import threading, time

NUM_THREADS = 3

def print_rt(a, b):
    print "I am %d" % a
    time.sleep(1)
    print "2 times me is %d" % b

def main():
    threads = [threading.Thread(target=print_rt, args=(i, 2 * i)) for i in range(NUM_THREADS)]
    for t in threads:
        t.start()

    for t in threads:
        t.join()

main()
