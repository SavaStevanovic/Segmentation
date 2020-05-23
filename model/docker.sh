docker build -t pytorch2001segplayground .
xhost + 
docker run -e DISPLAY=$DISPLAY --ipc=host --gpus all -p 6005:6005 -p 5001:5001 -it -v `pwd`/project:/app -v `pwd`/../common:/common -v /media/Data/CarSegmentation:/Data/ -v /tmp/.X11-unix:/tmp/.X11-unix pytorch2001segplayground
