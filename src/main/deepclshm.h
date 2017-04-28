#ifdef __cplusplus
extern "C" {
#endif

#ifndef QUERY_SZ
#error QUERY_SZ not defined
#endif

#ifndef RESPONSE_SZ
#error RESPONSE_SZ not defined
#endif

enum { SHM_ON, SHM_OFF, SHM_COMPUTING };

#include <sys/types.h>

/* Data model:
 * struct shmHeader at the beginning of the memory segment
 * followed by struct shmQuery instances until the end of the memory segment
 * working as a ring buffer.
 * - The client checks that queryReady is cleared for current client item
 *   then writes the query and sets the flag queryReady
 * - The server checks that queryReady is set for current server item
 *   then computes and writes the response, then sets reponseReady and clears queryReady
 * - The client checks that reponseReady is set, reads the response and clears reponseReady
 */

typedef struct {
	int querySize; /*server only*/
	int reponseSize; /*server only*/
	char netDef[1024]; /*server only*/
	char weightFile[1024]; /*server only*/
	int terminate; /*client sets, server clears*/
	} shmHeader;

typedef struct {
	int queryReady; /*client sets, server clears*/
	int reponseReady; /*server sets, client clears*/
	float query[QUERY_SZ]; /*client only*/
	float reponse[RESPONSE_SZ]; /*server only*/
	} shmQuery;

#ifdef __cplusplus
}
#endif
