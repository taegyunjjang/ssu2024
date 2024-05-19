#include "types.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "mmu.h"
#include "spinlock.h"
#include "slab.h"

struct {
	struct spinlock lock;
	struct slab slab[NSLAB];
} stable;

int slab_size[NSLAB] = {16, 32, 64, 128, 256, 512, 1024, 2048};

void slabinit(){
	initlock(&stable.lock, "slab");

	for (int i = 0; i < NSLAB; i++) {
		struct  slab *s = &stable.slab[i];

		s->size = slab_size[i];
		s->num_pages = 1;
		s->num_free_objects = PGSIZE / s->size;
		s->num_used_objects = 0;
		s->num_objects_per_page = PGSIZE / s->size;

		// Allocate one page for bitmap
		s->bitmap = kalloc();
		if (s->bitmap == 0) {
    		panic("slabinit: failed to allocate one page for bitmap");
		}
		memset(s->bitmap, 0, PGSIZE);

		// Allocate one page for slab cache
		s->page[0] = kalloc();
		if (s->page[0] == 0) {
			kfree(s->bitmap);
			panic("slabinit: failed to allocate one page for slab cache");
		}
		for (int i = 1; i < MAX_PAGES_PER_SLAB; i++) {
			s->page[i] = 0;
		}
	}
}

char *kmalloc(int size){
	acquire(&stable.lock);

	struct slab *s = 0;

	for (int i = 0; i < NSLAB; i++) {
		if (size <= slab_size[i]) {
			s = &stable.slab[i];
			break;
		}
	}

	if (s->num_free_objects == 0) {
		if (s->num_pages >= MAX_PAGES_PER_SLAB) {
			release(&stable.lock);
			return 0;
		}

		char *alloc_page = kalloc();
		if (alloc_page == 0) {
			panic("kmalloc: failed to allocate page");
		}

		int page_idx = s->num_pages;
		s->page[page_idx] = alloc_page;

		s->num_pages++;
		s->num_free_objects += s->num_objects_per_page;

		s->bitmap[(page_idx * s->num_objects_per_page) / 8] |= (1 << (page_idx * s->num_objects_per_page) % 8);
        s->num_free_objects--;
        s->num_used_objects++;
        char *obj = s->page[page_idx];
        
        release(&stable.lock);
        return obj;
	}

	for (int i = 0; i < s->num_objects_per_page * s->num_pages; i++) {
		if ((s->bitmap[i / 8] & (1 << (i % 8))) == 0) {
			s->bitmap[i / 8] |= (1 << (i % 8));
            s->num_free_objects--;
            s->num_used_objects++;

            int page_idx = i / s->num_objects_per_page;
            int offset = i % s->num_objects_per_page;
            char *obj = s->page[page_idx] + offset * s->size;

            release(&stable.lock);
            return obj;
		}
	}
	return 0;
}

void kmfree(char *addr, int size){
	acquire(&stable.lock);

	struct slab *s = 0;

	for (int i = 0; i < NSLAB; i++) {
		if (size <= slab_size[i]) {
			s = &stable.slab[i];
			break;
		}
	}

	if (s == 0) {
        release(&stable.lock);
        panic("kmfree: invalid size");
    }


	int page_idx = -1;
	for (int i = 0; i < s->num_pages; i++) {
		if (addr >= s->page[i] && addr < s->page[i] + PGSIZE) {
			page_idx = i;
			break;
		}
	}

	int objects = s->num_objects_per_page;
	int offset = (addr - s->page[page_idx]) / s->size;
    int idx = page_idx * objects + offset;

    if ((s->bitmap[idx / 8] & (1 << (idx % 8))) == 0) {
        release(&stable.lock);
        panic("kmfree: double free or invalid address");
    }

    s->bitmap[idx / 8] &= ~(1 << (idx % 8));
    s->num_free_objects++;
    s->num_used_objects--;

	/*
	// Page deallocation, if all slabs in each page is released 
    int page_dealloc = 1;
    for (int i = page_idx * objects; i < (page_idx + 1) * objects; i++) {
        if (s->bitmap[i / 8] & (1 << (i % 8))) {
            page_dealloc = 0;
            break;
        }
    }

    if (page_dealloc) {
        kfree(s->page[page_idx]);
        s->page[page_idx] = 0;
        s->num_pages--;

        for (int i = page_idx; i < s->num_pages; i++) {
            s->page[i] = s->page[i + 1];
        }

        for (int i = page_idx * objects; i < (s->num_pages + 1) * objects; i++) {
            s->bitmap[i / 8] = s->bitmap[(i + objects) / 8];
        }
    }
	*/
	
	release(&stable.lock);
}

/* Helper functions */
void slabdump(){
	cprintf("__slabdump__\n");

	struct slab *s;

	cprintf("size\tnum_pages\tused_objects\tfree_objects\tbitmap\n");

	for(s = stable.slab; s < &stable.slab[NSLAB]; s++){
		cprintf("%d\t%d\t\t%d\t\t%d\t\t%d\n", 
			s->size, s->num_pages, s->num_used_objects, s->num_free_objects, s->bitmap[0]);
	}
}

int numobj_slab(int slabid)
{
	return stable.slab[slabid].num_used_objects;
}

int numpage_slab(int slabid)
{
	return stable.slab[slabid].num_pages;
}
