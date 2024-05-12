import argparse
import os
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler

import glob

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/data/supasorn/dwui/stable-diffusion-webui/bin')

args = parser.parse_args()

# watch folder path for new files, then write to pipe /tmp/my_pipe
class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        # if not png return
        if not event.src_path.endswith(".png"):
            return

        print("Got event: ", event)
        print("Got event path: ", event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return

        if not event.dest_path.endswith(".png"):
            return

        print("Got moved event: ", event)
        print("Got moved event src path: ", event.src_path)
        print("Got moved event dest path: ", event.dest_path)
        # copy the file to /data/supasorn/img3dviewer/images
        os.system("cp " + event.dest_path + " /data/supasorn/img3dviewer/images")
        os.system("rm " + event.dest_path)
        with open("/tmp/my_pipe", "w", os.O_NONBLOCK) as f:
            f.write("watch dog\n")


def watch_folder(path):
    print("Watching folder: ", path)
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        observer.join()  # This waits for the observer to stop
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

if __name__ == '__main__':
    print(args.path)
    watch_folder(args.path)

    
