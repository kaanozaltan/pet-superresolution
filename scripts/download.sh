#!/bin/bash

username=""
password=""
login_url=""
download_url=""
output_filename=""

session_cookie=$(curl -s -c cookie.txt -d "username=$username&password=$password" $login_url | grep "session_cookie=" | cut -d "=" -f 2)
curl -s -b cookie.txt -o "$output_filename" $download_url
rm cookie.txt
