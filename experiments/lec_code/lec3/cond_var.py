import threading, time

NUM_THREADS = 3
TCOUNT = 10
COUNT_LIMIT = 12

count_mutex = threading.Lock()
threshold_condvar = threading.Condition(count_mutex)

def inc_count(t, count):
    my_id = t
    for i in range(TCOUNT):
        threshold_condvar.acquire()
        count[0] += 1

        if count[0] == COUNT_LIMIT:
            threshold_condvar.notify()
            print "inc_count(): thread %ld, count = %d. Threshold reached... Signal was sent." \
                    % (my_id, count[0])
        else:
            print "inc_count(): thread %ld, count = %d." % (my_id, count[0])

        threshold_condvar.release()
        time.sleep(1)

def watch_count(t, count):
    my_id = t
    print "watch_count(): thread %ld. Waiting on condition..." % my_id

    threshold_condvar.acquire()
    threshold_condvar.wait()
    count[0] += 125;
    print "watch_count(): thread %ld. Signal received. Added 125 to count = %d" % (my_id, count[0])
    threshold_condvar.release()


def main():
    ts = [1, 2, 3]
    count = [0]
    threads = [threading.Thread(target=watch_count, args=(ts[0], count))]
    threads.extend([threading.Thread(target=inc_count, args=(ts[i+1], count)) for i in range(NUM_THREADS - 1)])

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print "Main(): waited and joined with %d threads. Final value of count = %d. Done." \
            % (NUM_THREADS, count[0])

main()
