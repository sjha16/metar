"""
queue_manager.py - Manages concurrent users without hitting API limits using a thread-safe synchronous queue.
"""

import time
import threading
from collections import deque

class RequestQueue:
    """
    Queue requests when API limits are near.
    Blocks threads in a FIFO queue so they execute sequentially rather than raising 429 rate limit errors.
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.queue = deque()
        self.processing = 0
        self.max_concurrent = max_concurrent
        self.lock = threading.Lock()
        self.total_processed = 0
        self.total_queued = 0
    
    def process_request(self, func, *args, **kwargs):
        """
        Process a request, queuing if active slots exceed max_concurrent.
        Blocks thread until its turn, then executes.
        """
        event = None
        with self.lock:
            if self.processing < self.max_concurrent:
                self.processing += 1
                self.total_processed += 1
            else:
                self.total_queued += 1
                event = threading.Event()
                self.queue.append(event)
        
        if event:
            # Block the calling thread (e.g. Streamlit worker) until its turn
            print(f"⏳ Request queue active. Waiting in position {len(self.queue)}...")
            event.wait()
            
            # Woken up! Acquire slot
            with self.lock:
                self.processing += 1
                self.total_processed += 1
        
        try:
            # Execute actual function
            result = func(*args, **kwargs)
            return result
        finally:
            with self.lock:
                self.processing -= 1
                # Wake up the next in line
                if self.queue:
                    next_event = self.queue.popleft()
                    next_event.set()
    
    def get_stats(self) -> dict:
        """Get queue statistics"""
        with self.lock:
            return {
                "queue_length": len(self.queue),
                "currently_processing": self.processing,
                "total_processed": self.total_processed,
                "total_queued": self.total_queued,
                "max_concurrent": self.max_concurrent
            }

# Global queue instance (default: max 5 concurrent requests)
request_queue = RequestQueue(max_concurrent=5)
