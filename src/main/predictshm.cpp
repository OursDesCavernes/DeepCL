// Copyright Hugh Perkins (hughperkins at gmail), Josef Moudrik 2015
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.


#include "DeepCL.h"
#include "loss/SoftMaxLayer.h"
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <chrono>
#include <thread>

#define QUERY_SZ 2888
#define RESPONSE_SZ 361
#include "deepclshm.h"

#ifdef _WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif // _WIN32
#include "clblas/ClBlasInstance.h"

using namespace std;

/* [[[cog
    # These are used in the later cog sections in this file:
    options = [
        {'name': 'gpuIndex', 'type': 'int', 'description': 'gpu device index; default value is gpu if present, cpu otw.', 'default': -1, 'ispublicapi': True},

        {'name': 'weightsFile', 'type': 'string', 'description': 'file to read weights from', 'default': 'weights.dat', 'ispublicapi': True},
        # removing loadondemand for now, let's always load exactly one batch at a time for now
        # ('loadOnDemand', 'int', 'load data on demand [1|0]', 0, [0,1], True},
        {'name': 'batchSize', 'type': 'int', 'description': 'batch size', 'default': 128, 'ispublicapi': True},

        # lets go with pipe for now, and then somehow shoehorn files in later?
        {'name': 'shmKey',  'type': 'key_t', 'description': 'SHM key', 'default': '666'},
        {'name': 'shmBufferLen',  'type': 'int', 'description': 'Number of item in buffer', 'default': '1000'},
        {'name': 'outputLayer', 'type': 'int', 'description': 'layer to write output from, default -1 means: last layer', 'default': -1},
        {'name': 'writeLabels', 'type': 'int', 'description': 'write integer labels, instead of probabilities etc (default 0)', 'default': 0},
        {'name': 'outputFormat', 'type': 'string', 'description': 'output format [binary|text]', 'default': 'text'}
    ]
*///]]]
// [[[end]]]

class Config {
public:
    /* [[[cog
        cog.outl('// generated using cog:')
        for option in options:
            cog.outl(option['type'] + ' ' + option['name'] + ';')
    */// ]]]
    // generated using cog:
    int gpuIndex;
    string weightsFile;
    int batchSize;
    key_t shmKey;
    int shmBufferLen;
    int outputLayer;
    int writeLabels;
    string outputFormat;
    // [[[end]]]

    Config() {
        /* [[[cog
            cog.outl('// generated using cog:')
            for option in options:
                defaultString = ''
                default = option['default']
                type = option['type']
                if type == 'string':
                    defaultString = '"' + default + '"'
                elif type == 'int':
                    defaultString = str(default)
                elif type == 'float':
                    defaultString = str(default)
                    if '.' not in defaultString:
                        defaultString += '.0'
                    defaultString += 'f'
                cog.outl(option['name'] + ' = ' + defaultString + ';')
        */// ]]]
        // generated using cog:
        gpuIndex = -1;
        weightsFile = "weights.dat";
        batchSize = 128;
	shmKey = 666;
	shmBufferLen = 1000;
        outputLayer = -1;
        writeLabels = 0;
        outputFormat = "text";
        // [[[end]]]
    }
};

void go(Config config) {
    bool verbose = true;

    int numPlanes;
    int imageSize;
    int i;
//    int imageSizeCheck;

//    int dims[3];
//    cin.read(reinterpret_cast< char * >(dims), 3 * 4l);
    numPlanes = 8;
    imageSize = 19;
//    imageSizeCheck = dims[2];
//    if(imageSize != imageSizeCheck) {
//        throw std::runtime_error("imageSize doesnt match imageSizeCheck, image not square");
//    }
    
    if(verbose) cout << "Planes: " << numPlanes << ", size: " << imageSize << endl;
    const long inputCubeSize = QUERY_SZ;

    //
    // ## Set up the Network
    //

    EasyCL *cl = 0;
    if(config.gpuIndex >= 0) {
        cl = EasyCL::createForIndexedGpu(config.gpuIndex, verbose);
    } else {
        cl = EasyCL::createForFirstGpuOtherwiseCpu(verbose);
    }
    ClBlasInstance blasInstance;

    NeuralNet *net;
    net = new NeuralNet(cl);

    // just use the default for net creation, weights are overriden from the weightsFile
    WeightsInitializer *weightsInitializer = new OriginalInitializer();

    if(config.weightsFile == "") {
        cout << "weightsFile not specified" << endl;
        return;
    }

    string netDef;
    if (!WeightsPersister::loadConfigString(config.weightsFile, netDef) ){
        cout << "Cannot load network definition from weightsFile." << endl;
        return;
    }
//    cout << "net def from weights file: " << netDef << endl;

    net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
    net->addLayer(NormalizationLayerMaker::instance()->translate(0.0f)->scale(1.0f) ); // This will be read from weights file

    if(!NetdefToNet::createNetFromNetdef(net, netDef, weightsInitializer) ) {
        return;
    }

    // ignored int and float, s.t. we can use loadWeights
    int ignI;
    float ignF;

    // weights file contains normalization layer parameters as 'weights' now.  We should probably rename weights to parameters
    // sooner or later ,but anyway, tehcnically, works for onw
    if(!WeightsPersister::loadWeights(config.weightsFile, string("netDef=")+netDef, net, &ignI, &ignI, &ignF, &ignI, &ignF) ){
        cout << "Cannot load network weights from weightsFile." << endl;
        return;
    }

    if(verbose) {
        net->print();
    }
    net->setBatchSize(config.batchSize);
    if(verbose) cout << "batchSize: " << config.batchSize << endl;

    //
    // ## Init SHM
    //

    int shmid;
    void *shm;

    if ((shmid = shmget(config.shmKey, sizeof(shmQuery) * config.shmBufferLen + sizeof(shmHeader) + 1, IPC_CREAT | 0600)) < 0) {
        cout << "Cannot load create SHM with shmget." << endl;
        return;
    }
    
    if ((shm = shmat(shmid, NULL, 0)) == (char *) -1) {
        cout << "Cannot attach SHM." << endl;
        return;
    }

    shmHeader *header = (shmHeader*)shm;

    header->querySize = QUERY_SZ;
    header->reponseSize = RESPONSE_SZ;
    std::size_t len = netDef.copy(header->netDef,1024,0);
    header->netDef[len] = '\0';
    len = config.weightsFile.copy(header->weightFile,1024,0);
    header->weightFile[len] = '\0';
    header->terminate = SHM_OFF;

    shmQuery *queryList = (shmQuery*)(shm + sizeof(shmHeader));

    for(i=0;i<config.shmBufferLen;i++)
    {
        queryList[i].queryReady = SHM_OFF;
        queryList[i].reponseReady= SHM_OFF;
//	memset(&queryList[i].query[0],0,QUERY_SZ);
    }
    i = 0;
    //
    // ## All is set up now
    //

    float *inputData = new float[ inputCubeSize * config.batchSize];

//    int *labels = new int[config.batchSize];
    int n = 0;
    int j = 0;
    int k;
    bool run = true;
    bool sleep = false;

    if(config.outputLayer == -1) {
        config.outputLayer = net->getNumLayers() - 1;
    }

//    cin.read(reinterpret_cast< char * >(inputData), inputCubeSize * config.batchSize * 4l);

    while(run) {
        // no point in forwarding through all, so forward through each, one by one
        if(config.outputLayer < 0 || config.outputLayer > net->getNumLayers()) {
            throw runtime_error("outputLayer should be the layer number of one of the layers in the network");
        }

	if(header->terminate == SHM_ON)
		break;
	k = i;
	for(j=0;j<config.batchSize;j++)
	{
		if(queryList[i].queryReady == SHM_ON )
		{
			memcpy(&inputData[j*inputCubeSize],&queryList[i].query[0],inputCubeSize * sizeof(float));
			queryList[i].reponseReady = SHM_COMPUTING;
			i++;
		}
		else
		{
			sleep = true;
			memset(&inputData[j*inputCubeSize],0,inputCubeSize);
		}
		if(i>=config.shmBufferLen)
			i=0;
	}

        dynamic_cast<InputLayer *>(net->getLayer(0))->in(inputData);
        for(int layerId = 0; layerId <= config.outputLayer; layerId++) {
            StatefulTimer::setPrefix("layer" + toString(layerId) + " ");
            net->getLayer(layerId)->forward();
            StatefulTimer::setPrefix("");
        }
	
	float const*output = net->getLayer(config.outputLayer)->getOutput();
	for(j=0;j<config.batchSize;j++)
	{
		if(queryList[k].reponseReady == SHM_COMPUTING)
		{
			memcpy(&queryList[k].reponse[0],output,1);
			queryList[k].reponseReady = SHM_ON;
			queryList[k].queryReady = SHM_OFF;
			k++;
			n++;
		}
	}
        if(sleep)
        {
            sleep=false;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    delete[] inputData;
//    delete[] labels;
    delete weightsInitializer;
    delete net;
    delete cl;
}

void printUsage(char *argv[], Config config) {
    cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
    cout << endl;
    cout << "Possible key=value pairs:" << endl;
    /* [[[cog
        cog.outl('// generated using cog:')
        cog.outl('cout << "public api, shouldnt change within major version:" << endl;')
        for option in options:
            name = option['name']
            description = option['description']
            if 'ispublicapi' in option and option['ispublicapi']:
                cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
        cog.outl('cout << "" << endl; ')
        cog.outl('cout << "unstable, might change within major version:" << endl; ')
        for option in options:
            if 'ispublicapi' not in option or not option['ispublicapi']:
                name = option['name']
                description = option['description']
                cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
    *///]]]
    // generated using cog:
    cout << "public api, shouldnt change within major version:" << endl;
    cout << "    gpuindex=[gpu device index; default value is gpu if present, cpu otw.] (" << config.gpuIndex << ")" << endl;
    cout << "    weightsfile=[file to read weights from] (" << config.weightsFile << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "" << endl; 
    cout << "unstable, might change within major version:" << endl; 
    cout << "    shmkey=[SHM key to use] (" << config.shmKey << ")" << endl;
    cout << "    outputlayer=[layer ddto write output from, default -1 means: last layer] (" << config.outputLayer << ")" << endl;
    cout << "    writelabels=[write integer labels, instead of probabilities etc (default 0)] (" << config.writeLabels << ")" << endl;
    cout << "    outputformat=[output format [binary|text]] (" << config.outputFormat << ")" << endl;
    // [[[end]]]
}

int main(int argc, char *argv[]) {
    Config config;
    if(argc == 2 && (string(argv[1]) == "--help" || string(argv[1]) == "--?" || string(argv[1]) == "-?" || string(argv[1]) == "-h") ) {
        printUsage(argv, config);
    }
    for(int i = 1; i < argc; i++) {
        vector<string> splitkeyval = split(argv[i], "=");
        if(splitkeyval.size() != 2) {
          cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
          exit(1);
        } else {
            string key = splitkeyval[0];
            string value = splitkeyval[1];
//            cout << "key [" << key << "]" << endl;
            /* [[[cog
                cog.outl('// generated using cog:')
                cog.outl('if(false) {')
                for option in options:
                    name = option['name']
                    type = option['type']
                    cog.outl('} else if(key == "' + name.lower() + '") {')
                    converter = '';
                    if type == 'int':
                        converter = 'atoi';
                    elif type == 'float':
                        converter = 'atof';
                    cog.outl('    config.' + name + ' = ' + converter + '(value);')
            */// ]]]
            // generated using cog:
            if(false) {
            } else if(key == "gpuindex") {
                config.gpuIndex = atoi(value);
            } else if(key == "weightsfile") {
                config.weightsFile = (value);
            } else if(key == "batchsize") {
                config.batchSize = atoi(value);
            } else if(key == "shmkey") {
                config.shmKey = atoi(value);
            } else if(key == "outputlayer") {
                config.outputLayer = atoi(value);
            } else if(key == "writelabels") {
                config.writeLabels = atoi(value);
            } else if(key == "outputformat") {
                config.outputFormat = (value);
            // [[[end]]]
            } else {
                cout << endl;
                cout << "Error: key '" << key << "' not recognised" << endl;
                cout << endl;
                printUsage(argv, config);
                cout << endl;
                return -1;
            }
        }
    }
    if(config.outputFormat != "text" && config.outputFormat != "binary") {
        cout << endl;
        cout << "outputformat must be 'text' or 'binary'" << endl;
        cout << endl;
        return -1;
    }
    try {
        go(config);
    } catch(runtime_error e) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }
}


