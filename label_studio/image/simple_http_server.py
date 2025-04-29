#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler, test
import sys
import socket

class CORSRequestHandler (SimpleHTTPRequestHandler):
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)

if __name__ == '__main__':
    port = 8082
    try:
        # Check if port is available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', port))
        sock.close()
        
        print(f"Starting server on port {port}")
        test(CORSRequestHandler, HTTPServer, port=port)
    except PermissionError:
        print(f"Error: Permission denied for port {port}. Try running as administrator or use a different port.")
    except OSError as e:
        print(f"Error: Port {port} is already in use. Try a different port.")
    except Exception as e:
        print(f"Error: {str(e)}")