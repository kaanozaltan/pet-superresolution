#!/bin/bash

username="kaan.ozaltan@ug.bilkent.edu.tr"
password=""
login_url="https://ida.loni.usc.edu/login.jsp"
download_url="https://downloads.loni.usc.edu/download/files/ida1/75f27461-05ab-4165-8cf4-80730ae2719c/ADNI1:Baseline%203T.zip"
output_filename="baseline.zip"

session_cookie=$(curl -s -c cookie.txt -d "username=$username&password=$password" $login_url | grep "session_cookie=" | cut -d "=" -f 2)
curl -s -b cookie.txt -o "$output_filename" $download_url
rm cookie.txt
