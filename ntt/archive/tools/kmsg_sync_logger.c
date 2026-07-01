/* kmsg_sync_logger.c — durably capture kernel messages up to a hard hang.
 *
 * Reads /dev/kmsg and writes each record to a logfile opened O_SYNC, fdatasync'ing
 * after every record, so the last kernel line before a total freeze survives on
 * disk for post-reboot analysis. Defeats journald's buffer-lost-on-hang problem.
 *
 * Robustness for sub-second freezes:
 *  - poll(): wakes the instant a new kernel record appears (no fixed poll delay).
 *  - SCHED_FIFO real-time priority: preempts the compute load so the fault line is
 *    written before the CPU wedges on its next GPU MMIO.
 *  - mlockall(): never paged out, so capture never waits on the disk for itself.
 *  - O_SYNC + fdatasync: each record is on the SSD before we read the next.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <poll.h>
#include <sched.h>
#include <sys/mman.h>

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <logfile>\n", argv[0]); return 2; }

    /* Lock memory so capture is never delayed by a page fault. We deliberately do
     * NOT raise to SCHED_FIFO real-time priority: an RT task that fails to yield
     * can starve kernel threads and hang the machine (self-inflicted 2026-06-12).
     * A slightly-niced normal task plus mlockall + poll() is fast enough to catch
     * the last kernel line before a ~1s flip-timeout window, without that risk. */
    (void)mlockall(MCL_CURRENT | MCL_FUTURE);
    int nr = nice(-10); (void)nr;   /* best-effort priority bump; never fatal */

    int k = open("/dev/kmsg", O_RDONLY | O_NONBLOCK);
    if (k < 0) { perror("open /dev/kmsg"); return 1; }
    lseek(k, 0, SEEK_END);                 /* only messages from now on */

    int f = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC | O_SYNC, 0644);
    if (f < 0) { perror("open logfile"); return 1; }

    char buf[8192];
    struct pollfd pfd = { .fd = k, .events = POLLIN };
    for (;;) {
        int pr = poll(&pfd, 1, 1000);      /* instant wake on new record */
        if (pr < 0 && errno == EINTR) continue;
        ssize_t n;
        while ((n = read(k, buf, sizeof buf)) > 0) {
            ssize_t off = 0;
            while (off < n) {
                ssize_t w = write(f, buf + off, (size_t)(n - off));
                if (w <= 0) break;
                off += w;
            }
            fdatasync(f);
        }
        /* n<0: EAGAIN (drained) or EPIPE (ring overrun) — re-poll. */
    }
    return 0;
}
