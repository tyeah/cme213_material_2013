#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_mutex_t mutext[2];

void* routine(void* t) {
    int i = (int)t;
    pthread_mutex_lock(&mutex[i]);
    printf("I am subroutine %d", i);
    pthread_mutex_unlock(&mutex[i]);


