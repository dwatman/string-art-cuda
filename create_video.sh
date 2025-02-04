ffmpeg -framerate 10 -i ./video/img_%04d.png -c:v libx264 -profile:v baseline -pix_fmt yuv420p output.mp4
