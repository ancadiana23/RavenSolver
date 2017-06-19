#!/bin/bash

rm resized/*
cp ./*/*.PNG resized/
for file in resized/*.PNG
do	
	convert -resize 550 $file $file
done

