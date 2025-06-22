# Prog 1
``` c++
int height = args->height, n = args->numThreads, i = args->threadId;
    int startRow = i * (height/n);
    int rows = args->height / args->numThreads;

    // if there is less than `rows` left in to compute, compute the rest of the image instead
    if(height - startRow < rows) {
        rows = height - startRow;
    }
    
    mandelbrotSerial(
        args->x0, args->y0, args->x1, args->y1,
        args->width, args->height,
        startRow, rows,
        args->maxIterations,
        args->output
);
```

Speedups
-----
\# of Threads
|  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |
|------|------|------|------|------|------|------|------|
| 0.97 | 1.89 | 1.59 | 2.31 | 2.38 | 2.98 | ERR  | 3.61 |

![alt text](image.png)

This seems to be a linear speedup (in terms of the factor). In terms of actual time, it would end up following a $1/x$ type function. It seems that odd numbered threads slow down rather than speed up the process.

Re-mapping
-----
Thread workloads were remapped such that instead of each thread being assigned a contiguous chunk of rows, each thread is cycled through on each row (i.e. thread 1 gets rows 0, 8, 16, and thread 2 gets rows 1, 9, 17, etc...).

16 threads does not speed up the program significantly; this is likely due to the fact that the computer only has 8 threads to run on, and as such 16 will just be adding more threads when there is not the execution capacity for them.