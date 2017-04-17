/*
 * Remarks:
 * Remember to do: pthread_mutex_init(&mutexsum, NULL);
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 3000
#define NUM_THREADS 3
#define USE_MUTEX 1

pthread_mutex_t mutexsum;

typedef struct {
    double* a;
    double* b;
    double* sum;
    double start;
} DOTDATA;

void print_dot(DOTDATA* d) {
    double s = 0;
    for (int i=0; i < VECLEN; i++) {
        s += d->a[i] * d->b[i];
    }
    printf("sum: %.3f\t%.3f\n", *d->sum, s);
}

void init_vec(double* v, double p, double q) {
    for (int i=0; i < VECLEN; i++) {
        v[i] = p * i + q;
    }
}

void* dodot(void* dotdata) {
    DOTDATA* d = (DOTDATA*)dotdata;
    for (int i = d->start; i < d->start + (VECLEN / NUM_THREADS); i++) {
        if (USE_MUTEX) {
            pthread_mutex_lock(&mutexsum);
        }
        *d->sum += d->a[i] * d->b[i];
        if (USE_MUTEX) {
            pthread_mutex_unlock(&mutexsum);
        }
    }
    pthread_exit(NULL);
}

int main() {
    double a[VECLEN], b[VECLEN], sum = 0;
    init_vec(a, 3, -2);
    init_vec(b, -2, 5);
    pthread_t threads[NUM_THREADS];
    pthread_mutex_init(&mutexsum, NULL);
    int rc;
    for (int t=0; t < NUM_THREADS; t++) {
        DOTDATA* d = (DOTDATA*)malloc(sizeof(DOTDATA));
        d->a = a;
        d->b = b;
        d->start = t * (VECLEN / NUM_THREADS);
        d->sum = &sum;
        pthread_create(&threads[t], NULL, dodot, (void*)d);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }
    for (int t=0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
    DOTDATA* d = (DOTDATA*)malloc(sizeof(DOTDATA));
    d->a = a;
    d->b = b;
    d->sum = &sum;
    print_dot(d);
    pthread_mutex_destroy(&mutexsum);
}


