#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <cstdlib>

#define MACRO_STRINGIFY(x) #x

unsigned char getRand(float mean, float std) {
    unsigned char randval;

    std::random_device rd;
    // Create and seed the generator
    std::mt19937 gen(rd());
    // Create distribution
    std::normal_distribution<> d(mean, std);
    // Generate random numbers according to distribution

    randval = std::round(d(gen));
    if (randval > 255)
        randval = 255;
    else if (randval < 0)
        randval = 0;
    return randval;
}

// RUNNING MEDIAN CODE START
typedef float Item;
typedef struct Mediator_t {
    Item* data;   // circular queue of values
    int*  pos;    // index into `heap` for each value
    int*  heap;   // max/median/min heap holding indexes into `data`.
    int   N;      // allocated size.
    int   idx;    // position in circular queue
    int   minCt;  // count of items in min heap
    int   maxCt;  // count of items in max heap
} Mediator;


/*--- Helper Functions ---*/

// returns 1 if heap[i] < heap[j]
inline int mmless(Mediator* m, int i, int j) {
    return (m->data[m->heap[i]] < m->data[m->heap[j]]);
}

// swaps items i&j in heap, maintains indexes
int mmexchange(Mediator* m, int i, int j) {
    int t              = m->heap[i];
    m->heap[i]         = m->heap[j];
    m->heap[j]         = t;
    m->pos[m->heap[i]] = i;
    m->pos[m->heap[j]] = j;
    return 1;
}

// swaps items i&j if i<j;  returns true if swapped
inline int mmCmpExch(Mediator* m, int i, int j) {
    return (mmless(m, i, j) && mmexchange(m, i, j));
}



// maintains minheap property for all items below i.
void minSortDown(Mediator* m, int i) {
    for (i *= 2; i <= m->minCt; i *= 2) {
        if (i < m->minCt && mmless(m, i + 1, i)) {
            ++i;
        }
        if (!mmCmpExch(m, i, i / 2)) {
            break;
        }
    }
}

// maintains maxheap property for all items below i. (negative indexes)
void maxSortDown(Mediator* m, int i) {
    for (i *= 2; i >= -m->maxCt; i *= 2) {
        if (i > -m->maxCt && mmless(m, i, i - 1)) {
            --i;
        }
        if (!mmCmpExch(m, i / 2, i)) {
            break;
        }
    }
}

// maintains minheap property for all items above i, including median
// returns true if median changed
inline int minSortUp(Mediator* m, int i) {
    while (i > 0 && mmCmpExch(m, i, i / 2))
        i /= 2;
    return (i == 0);
}

// maintains maxheap property for all items above i, including median
// returns true if median changed
inline int maxSortUp(Mediator* m, int i) {
    while (i < 0 && mmCmpExch(m, i / 2, i))
        i /= 2;
    return (i == 0);
}



/*--- Public Interface ---*/

// creates new Mediator: to calculate `nItems` running median.
// mallocs single block of memory, caller must free.
Mediator* MediatorNew(int nItems) {
    int size    = sizeof(Mediator) + nItems * (sizeof(Item) + sizeof(int) * 2);
    Mediator* m = (Mediator*)std::malloc(size);
    m->data     = (Item*)(m + 1);
    m->pos      = (int*)(m->data + nItems);
    m->heap  = m->pos + nItems + (nItems / 2);  // points to middle of storage.
    m->N     = nItems;
    m->minCt = m->maxCt = m->idx = 0;
    // set up initial heap fill pattern: median,max,min,max,...
    while (nItems--) {
        m->pos[nItems]          = ((nItems + 1) / 2) * ((nItems & 1) ? -1 : 1);
        m->heap[m->pos[nItems]] = nItems;
    }
    return m;
}

// Inserts item, maintains median in O(lg nItems)
void MediatorInsert(Mediator* m, Item v) {
    int  p          = m->pos[m->idx];
    Item old        = m->data[m->idx];
    m->data[m->idx] = v;
    m->idx          = (m->idx + 1) % m->N;
    if (p > 0) {
        // new item is in minHeap
        if (m->minCt < (m->N - 1) / 2) {
            m->minCt++;
        } else if (v > old) {
            minSortDown(m, p);
            return;
        }
        if (minSortUp(m, p) && mmCmpExch(m, 0, -1)) {
            maxSortDown(m, -1);
        }
    } else if (p < 0) {
        // new item is in maxheap
        if (m->maxCt < m->N / 2) {
            m->maxCt++;
        } else if (v < old) {
            maxSortDown(m, p);
            return;
        }
        if (maxSortUp(m, p) && m->minCt && mmCmpExch(m, 1, 0)) {
            minSortDown(m, 1);
        }
    } else {
        // new item is at median
        if (m->maxCt && maxSortUp(m, -1)) {
            maxSortDown(m, -1);
        }
        if (m->minCt && minSortUp(m, 1)) {
            minSortDown(m, 1);
        }
    }
}


// returns median item (or average of 2 when item count is even)
Item MediatorMedian(Mediator* m) {
    Item v = m->data[m->heap[0]];
    if (m->minCt < m->maxCt) {
        v = (v + m->data[m->heap[-1]]) / 2;
    }
    return v;
}
// RUNNING MEDIAN CODE END




#define ELEM_SWAP(a,b) { float t=(a);(a)=(b);(b)=t; }

float median(float arr[], int n) {
    int low, high;
    int median;
    int middle, ll, hh;

    low    = 0;
    high   = n - 1;
    median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median];

        if (high == low + 1) { /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]);
            return arr[median];
        }

        /* Find median of low, middle and high items; swap into position low */
        middle = (low + high) / 2;
        if (arr[middle] > arr[high])
            ELEM_SWAP(arr[middle], arr[high]);
        if (arr[low] > arr[high])
            ELEM_SWAP(arr[low], arr[high]);
        if (arr[middle] > arr[low])
            ELEM_SWAP(arr[middle], arr[low]);

        /* Swap low item (now in position middle) into position (low+1) */
        ELEM_SWAP(arr[middle], arr[low + 1]);

        /* Nibble from each end towards middle, swapping items when stuck */
        ll = low + 1;
        hh = high;
        for (;;) {
            do
                ll++;
            while (arr[low] > arr[ll]);
            do
                hh--;
            while (arr[hh] > arr[low]);

            if (hh < ll)
                break;

            ELEM_SWAP(arr[ll], arr[hh]);
        }

        /* Swap middle item (in position low) back into correct position */
        ELEM_SWAP(arr[low], arr[hh]);

        /* Re-set active partition */
        if (hh <= median)
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}
#undef ELEM_SWAP

