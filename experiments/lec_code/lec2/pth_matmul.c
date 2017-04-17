#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define MAT_DIM 3

typedef struct {
    int size, row, column;
    double (*MA)[MAT_DIM], (*MB)[MAT_DIM], (*MC)[MAT_DIM];
} matrix_type_t;

void* thread_mult (void* w) {
    matrix_type_t* work = (matrix_type_t*) w;
    int i = work->row;
    int j = work->column;
    work->MC[i][j] = 0;
    for (int k=0; k < work->size; k++) {
        work->MC[i][j] += work->MA[i][k] * work->MB[k][j];
    }
    pthread_exit(NULL);
}

void print_mat (double mat[][MAT_DIM]) {
    for (int i=0; i < MAT_DIM; i++) {
        for (int j=0; j < MAT_DIM; j++) {
            printf("%.3f\t", mat[i][j]);
        }
        printf("\n");
    }
}
void print_work (matrix_type_t* work) {
    printf("MA\n");
    print_mat(work->MA);
    printf("MB\n");
    print_mat(work->MB);
    printf("MC\n");
    print_mat(work->MC);
}

void init_mat (double mat[][MAT_DIM]) {
    for (int i=0; i < MAT_DIM; i++) {
        for (int j=0; j < MAT_DIM; j++) {
            mat[i][j] = (i + 1) * (j + 1);
        }
    }
}

int main(int argc, char* argv[]) {
    pthread_t threads[MAT_DIM][MAT_DIM];
    double MA[MAT_DIM][MAT_DIM], MB[MAT_DIM][MAT_DIM], MC[MAT_DIM][MAT_DIM];
    init_mat(MA);
    init_mat(MB);
    int rc;
    for (int i=0; i < MAT_DIM; i++) {
        for (int j=0; j < MAT_DIM; j++) {
            matrix_type_t* work = (matrix_type_t*)malloc(sizeof(matrix_type_t));
            work->row = i;
            work->column = j;
            work->size = MAT_DIM;
            work->MA = MA;
            work->MB = MB;
            work->MC = MC;
            rc = pthread_create(&threads[i][j], NULL, &thread_mult, (void*)work);
            if (rc) {
                printf("ERROR; return code from pthread_create() is %d\n", rc);
                exit(-1);
            }

        }
    }
    for (int i=0; i < MAT_DIM; i++) {
        for (int j=0; j < MAT_DIM; j++) {
            pthread_join(threads[i][j], NULL);
        }
    }
    matrix_type_t* work = (matrix_type_t*)malloc(sizeof(matrix_type_t));
    work->row = MAT_DIM;
    work->column = MAT_DIM;
    work->size = MAT_DIM;
    work->MA = MA;
    work->MB = MB;
    work->MC = MC;
    print_work(work);
}
