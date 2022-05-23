from BaseHTTPServer import BaseHTTPRequestHandler
import urlparse, json
import urllib
import os
import traceback
import subprocess

class GetHandler(BaseHTTPRequestHandler):

    def do_GET(self):
      try:
        parsed_path = urlparse.urlparse(self.path)
        #message = '\n'.join([
        #    'CLIENT VALUES:',
        #    'client_address=%s (%s)' % (self.client_address,
        #        self.address_string()),
        #    'command=%s' % self.command,
        #    'path=%s' % self.path,
        #    'real path=%s' % parsed_path.path,
        #    'query=%s' % parsed_path.query,
        #    'request_version=%s' % self.request_version,
        #    '',
        #    'SERVER VALUES:',
        #    'server_version=%s' % self.server_version,
        #    'sys_version=%s' % self.sys_version,
        #    'protocol_version=%s' % self.protocol_version,
        #    '',
        #    ])
        print 'request url is ', self.path
        uri = self.path[1:]
        params = {}
        for pair_str in uri.split('&'):
          pair = pair_str.split('=')
          if len(pair) != 2:
            raise Exception('key pair must be 2, but got ', len(pair))
          params[pair[0]] = pair[1]

        print 'params', params
        ptxas_path = params["ptxas_path"] if "ptxas_path" in params else None
        src_name   = params["src_name"] if "src_name" in params else None
        dst_name   = params["dst_name"] if "dst_name" in params else None
        disable_ptxas_optimizations   = params["disable_ptxas_optimizations"] \
                if "disable_ptxas_optimizations" in params else "false"
        arch       = params["arch"] if "arch" in params else None
        if ptxas_path is None or src_name is None or dst_name is None or arch is None:
          raise Exception('ptxas_path src_name dst_name and arch cannot be None')

        cmd = ptxas_path + " " + src_name + " -o " + dst_name + " -arch="+ arch
        if disable_ptxas_optimizations in ["true", "1"]:
            cmd += " -O0"
        print 'cmd is ', cmd
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
          raise Exception('cmd exec error, got ret {}, cmd {}'.format(ret, cmd))

        self.send_response(200)
        self.end_headers()
        self.wfile.write("success")
        return
      except Exception as e:
        print traceback.print_exc()
        self.send_response(400)
        self.end_headers()
        self.wfile.write(str(e))
        

    def do_POST(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write("success")
        return

if __name__ == '__main__':
    from BaseHTTPServer import HTTPServer
    port = os.getenv('TF_PTXAS_HTTP_PORT')
    if port is None:
        port = 8881
    server = HTTPServer(('localhost', port), GetHandler)
    print 'Starting server at http://localhost:8881'
    server.serve_forever()
