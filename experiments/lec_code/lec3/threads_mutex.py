import threading, time

NUM_THREADS = 3
mutex0 = threading.Lock()

def print_rt(a, b):
    mutex0.acquire()
    print "I am %d" % a
    time.sleep(1)
    print "Value of me is %d" % b
    mutex0.release()

def main():
    threads = [threading.Thread(target=print_rt, args=(i, 2 * i)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()

    for t in threads:
        t.join()

main()
