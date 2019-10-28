// g++ xshm2.c -o xshm2 -lX11 -lXext `$cv`-Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2 && ./xshm2

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "stdio.h"
#include <opencv2/opencv.hpp>  // This includes most headers!
#include <string>
#include <sstream>
#include <sys/time.h>
#include <time.h>
#define FPS(start) (CLOCKS_PER_SEC / (clock()-start))

// Using one monitor DOESN'T improve performance! Querying a smaller subset of the screen DOES
//const uint WIDTH  = 1920>>0;
//const uint HEIGHT = 1080>>0;
const uint WIDTH  = 800>>0;
const uint HEIGHT = 600>>0;

int getsc(){
    Display* display = XOpenDisplay(NULL);
    Window root = DefaultRootWindow(display);  // Macro to return the root window! It's a simple uint32
    
    if (!root){fprintf(stderr, "unable to connect to display");return 7;}

    XWindowAttributes window_attributes;
    XGetWindowAttributes(display, root, &window_attributes);
    Screen* screen = window_attributes.screen;
    XShmSegmentInfo shminfo;
    XImage* ximg = XShmCreateImage(display, DefaultVisualOfScreen(screen), DefaultDepthOfScreen(screen), ZPixmap, NULL, &shminfo, WIDTH, HEIGHT);

    shminfo.shmid = shmget(IPC_PRIVATE, ximg->bytes_per_line * ximg->height, IPC_CREAT|0777);
    shminfo.shmaddr = ximg->data = (char*)shmat(shminfo.shmid, 0, 0);
    shminfo.readOnly = False;
    if(shminfo.shmid < 0)
        puts("Fatal shminfo error!");;
    Status s1 = XShmAttach(display, &shminfo);
    printf("XShmAttach() %s\n", s1 ? "success!" : "failure!");


    cv::Mat img;	
    int it=0;
    int current = 0;
    std::string current_name=""; 

    // path de salida
    char path_salida[40] = "../images_buffer/";

    // array con nombres actuales
    std::string name_array[100];
    std::string full_path="";
    char t[60];
    std::string time_string="";

    for(int i; ; i++){
        double start = clock();
	
	// Obten imagen
        XShmGetImage(display, root, ximg, 0, 0, 0x00ffffff);
        img = cv::Mat(HEIGHT, WIDTH, CV_8UC4, ximg->data);
	//cv::resize(img, img, cv::Size(img.cols * 0.25,img.rows * 0.25), 0, 0, CV_INTER_LINEAR);
        //if(!(i & 0b111111))
        //    printf("fps %4.f  spf %.4f\n", FPS(start), 1 / FPS(start));
	
	// calcula timestamp
	timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	sprintf(t,"%d_%d",ts.tv_sec,ts.tv_nsec);
	time_string = t;
        	
 

	// calcula nombre salida num_timestamp.png
	current = it % 100;
	current_name = "";
	sprintf(t,"%d_%s.png", current,time_string.c_str());
	current_name = t;

	// borrar de disco archivo antiguo
	if (!name_array[current].empty()) {
		full_path = std::string(path_salida) + std::string(name_array[current]);
		remove(full_path.c_str());
	}
	// guarda nombre para borra despues
       	name_array[current] = current_name; 	

	// escribe imagen a archivo
	full_path = std::string(path_salida) + std::string(current_name);
	cv::imwrite(full_path,img);
	
	//std::cout << full_path << "\n";

	//if (it > 3000) {break;}
	it++;
	
	//cv::imshow("img", img);
	//char c = cv::waitKey(1);
	//if( c == 27 ) break;
    }


    XShmDetach(display, &shminfo);
    XDestroyImage(ximg);
    shmdt(shminfo.shmaddr);
    XCloseDisplay(display);
    puts("Exit success!");
    return 0;
}

// -------------------------------------------------------
int main(){
	return getsc();
}
