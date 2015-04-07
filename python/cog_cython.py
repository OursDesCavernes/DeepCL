# Copyright Hugh Perkins 2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

# functions to help wrap C++ callback classes in Cython, and more

import cog

def upperFirst( word ):
    """helper method to capitalize the first letter of word"""
    word = word[0].upper() + word[1:]
    return word

def cpp_write_proxy_class( proxy_name, parent_name, defs ):
    """use to create a c++ class that inherits from a (possibly abstract) c++ class
    and handles the c++ side of receiving callback functions into cython,
    and calling these appropriately"""

    cog.outl('// generated using cog (as far as the [[end]] bit:')
    cog.outl( 'class ' + proxy_name + ' : public ' + parent_name + ' {' )
    cog.outl( 'public:')
    cog.outl( '    void *pyObject;')
    cog.outl( '')
    cog.outl( '    CyScenario(void *pyObject) :')
    cog.outl( '        pyObject(pyObject) {')
    cog.outl( '    }')
    cog.outl( '')

    for thisdef in defs:
        ( name, returnType, parameters ) = thisdef
        cog.out('    typedef ' + returnType + '(*' + name + 'Def)(')
        for parameter in parameters:
            (ptype,pname) = parameter
            cog.out( ptype + ' ' + pname + ',')
        cog.outl( ' void *pyObject);')
    cog.outl('')

    for thisdef in defs:
        ( name, returnType, parameters ) = thisdef
        cog.outl( '    ' + name + 'Def c' + upperFirst( name ) + ';' )   
    cog.outl('')     

    for thisdef in defs:
        ( name, returnType, parameters ) = thisdef
        cog.outl( '    void set' + upperFirst( name ) + ' ( ' + name + 'Def c' + upperFirst( name ) + ' ) {' )   
        cog.outl( '        this->c' + upperFirst( name ) + ' = c' + upperFirst( name ) + ';' )
        cog.outl( '    }')
    cog.outl('')     

    for thisdef in defs:
        ( name, returnType, parameters ) = thisdef
        cog.out( '    virtual ' + returnType + ' ' + name + '(' )
        isFirstParam = True
        for param in parameters:
            (ptype,pname) = param
            if not isFirstParam:
                cog.out(', ')
            cog.out( ptype + ' ' + pname )
            isFirstParam = False
        cog.outl(') {')
        # cog.outl('    std::cout << "CyScenario.' + name + '()" << std::endl;')
        cog.out('        ')
        if returnType != 'void':
            cog.out('return ')
        cog.out('c' + upperFirst( name ) + '(')
        for param in parameters:
            (ptype,pname) = param
            cog.out( pname + ', ' )
        cog.outl( 'pyObject );' )
        cog.outl('    }')
    cog.outl( '};' )

def pxd_write_proxy_class( proxy_name, defs ):
    """writes the pxd declaration of the same class that was created using
    'cpp_write_proxy_class' for C++ above.
    This should be used inside 'cdef extern from "somefile.h":' section"""

    cog.outl('# generated using cog (as far as the [[end]] bit:')
    for thisdef in defs:
        ( name, returnType, parameters ) = thisdef
        cog.out('ctypedef ' + returnType + '(*' + proxy_name + '_' + name + 'Def)(')
        for parameter in parameters:
            (ptype,pname) = parameter
            cog.out( ptype + ' ' + pname + ',')
        cog.outl( ' void *pyObject)')

    cog.outl( 'cdef cppclass ' + proxy_name + ':')
    cog.outl( '    ' + proxy_name + '(void *pyObject)')
    cog.outl( '' )
    for thisdef in defs:
        ( name, returnType, parameters ) = thisdef
        cog.outl( '    void set' + upperFirst( name ) + ' ( ' + proxy_name + '_' + name + 'Def c' + upperFirst( name ) + ' )')

