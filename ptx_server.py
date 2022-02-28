#!/usr/bin/python
#****************************************************************#
# ScriptName: server1.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2022-02-26 11:14
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2022-02-28 09:54
# Function: 
#***************************************************************#
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import io,shutil,urllib,os
import requests
import time

data = {'result': 'this is a test'}
host = ('localhost', 8888)

def send(src_name, dst_name, arch):
    url = f"http://11.186.54.222"
    files = {
        "img": open(src_name, 'rb')
    }
    data = {
        "src_name": src_name,
        "dst_name": dst_name,
        "arch": arch,
    }

    response = requests.post(url, files=files, data=data)
    file = open(dst_name, 'wb')
    file.write(response.content)
    file.close()

class Resquest(BaseHTTPRequestHandler):
    def do_GET(self):
        print('==========')
        if '?' in self.path:#如果带有参数
            self.queryString=urllib.parse.unquote(self.path.split('?',1)[1])
            params=urllib.parse.parse_qs(self.queryString)
            print(params)
            src_name=params["src_name"][0] if "src_name" in params else None
            dst_name=params["dst_name"][0] if "dst_name" in params else None
            arch=params["arch"][0] if "arch" in params else None
            #send(src_name, dst_name, arch)
            os.system('/usr/local/cuda/bin/ptxas ' + src_name + " -o " + dst_name + " -arch="+ arch)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
