import threading, Queue, time

num_threads = 10
num_tasks = 30

def subroutine(q, i):
    while True:
        element = q.get()
        output = ("I am thread %d processing: (%d, %d)" % ((i,) + element))
        print output
        time.sleep(1)
        q.task_done()

def main():
    q = Queue.PriorityQueue()
    workers = []
    for i in range(num_threads):
        worker = threading.Thread(target=subroutine, args=(q, i))
        worker.setDaemon(True) # important!
        worker.start()
        workers.append(worker)

    for i in range(num_tasks):
        q.put((num_tasks - i, i))

    # q.join will let program waiting for all the tasks in q to be processed
    # used for sync
    q.join() 
    '''
    for w in workers:
        # w.join will will let program waiting for w to exit even if w is a daemon, 
        # but in this case it will never exit because of the infinite loop
        # if w is a daemon and w.join is not called, the program will exit without waiting for w to exit
        # if w is not a daemon and w.join is not called, the program will still wait w to exit
        w.join() 
    '''
    print "Finish"

main()
