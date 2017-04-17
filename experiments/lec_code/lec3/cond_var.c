#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // for sleep

#define NUM_THREADS     3
#define TCOUNT          10
#define COUNT_LIMIT     12

int count = 0;
pthread_mutex_t count_mutex;
pthread_cond_t threshold_condvar;

void* inc_count(void* t) {
    long my_id = (long)t;

    for (int i=0; i < TCOUNT; i++) {
        pthread_mutex_lock(&count_mutex);

        count++;

        /* Check the value of count and signal waiting thread when condition is
         * reached. */
        if (count == COUNT_LIMIT) {
            printf("inc_count(): thread %ld, count = %d. Threshold reached... ",
                    my_id, count);
            pthread_cond_signal(&threshold_condvar);
            printf("Signal was sent.\n");
        } else {
            printf("inc_count(): thread %ld, count = %d.\n",
                    my_id, count);
        }

        pthread_mutex_unlock(&count_mutex);

        /* Do some “work” */
        sleep(1);
    }
    pthread_exit(NULL);
}

void* watch_count(void* t) {
    long my_id = (long)t;

    printf("watch_count(): thread %ld. Waiting on condition...\n", my_id);

    pthread_mutex_lock(&count_mutex);
    pthread_cond_wait(&threshold_condvar, &count_mutex);
    /*
    while (count < COUNT_LIMIT) {
        pthread_cond_wait(&threshold_condvar, &count_mutex);
    }
    */

    count += 125;
    printf("watch_count(): thread %ld. Signal received.", my_id);
    printf(" Added 125 to count = %d\n", count);
    pthread_mutex_unlock(&count_mutex);
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    int i, rc;
    long t1=1, t2=2, t3=3;
    pthread_t threads[3];

    /* Initialize mutex and condition variable objects */
    pthread_mutex_init(&count_mutex, NULL);
    pthread_cond_init (&threshold_condvar, NULL);
   
    pthread_create(&threads[0], NULL, watch_count, (void *)t1);
    pthread_create(&threads[1], NULL, inc_count, (void *)t2);
    pthread_create(&threads[2], NULL, inc_count, (void *)t3);
   
    /* Wait for all threads to complete */
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Main(): waited and joined with %d threads. Final value of count = %d. Done.\n",
            NUM_THREADS, count);
   
    /* Clean up and exit */
    pthread_mutex_destroy(&count_mutex);
    pthread_cond_destroy(&threshold_condvar);
    pthread_exit (NULL);
}
