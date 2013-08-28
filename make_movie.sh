#!/bin/bash
echo "I'm gonna make you a star!"
rm ./images/fig_*
python generate_images.py
ffmpeg -i ./images/fig_%d.png -loop 5 -pix_fmt yuv420p particle_motion.mp4
open particle_motion.mp4
