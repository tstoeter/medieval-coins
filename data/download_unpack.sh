#!/usr/bin/bash

for url in $(cat urls.txt);
do
    wget -nc $url
done

unzip -u \*.zip

