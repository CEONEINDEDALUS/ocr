#!/usr/bin/env python3

import json
import http.server
import socketserver
from urllib.parse import urlparse
import sys
import os
import socket

# Import our OCR neural network
from ocr import OCRNeuralNetwork

class OCRHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests (serve HTML file for development)"""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Content-Length, Authorization')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS, PUT, DELETE')
            self.end_headers()
            
            try:
                with open('ocr.html', 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            except FileNotFoundError:
                self.wfile.write(b'HTML file not found. Please create ocr.html')
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Content-Length')
        self.end_headers()

    def do_POST(self):
        """Handle POST requests for training and prediction"""
        response_code = 200
        response = ""
        
        try:
            # Get content length and read the request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                response_code = 400
                response = {"error": "No data received"}
            else:
                content = self.rfile.read(content_length)
                payload = json.loads(content.decode('utf-8'))

                # Handle training requests
                if payload.get('train'):
                    print(f"Training with {len(payload['trainArray'])} samples...")
                    nn.train(payload['trainArray'])
                    nn.save()
                    response = {"status": "training_complete", "samples": len(payload['trainArray'])}
                    print("Training completed and weights saved.")

                # Handle prediction requests  
                elif payload.get('predict'):
                    print("Making prediction...")
                    try:
                        prediction = nn.predict(payload['image'])
                        response = {
                            "type": "test", 
                            "result": prediction
                        }
                        print(f"Prediction: {prediction}")
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        response_code = 500
                        response = {"error": "Prediction failed"}
                        
                # Invalid request
                else:
                    response_code = 400
                    response = {"error": "Invalid request. Use 'train' or 'predict' parameter."}

        except json.JSONDecodeError:
            response_code = 400
            response = {"error": "Invalid JSON"}
        except Exception as e:
            print(f"Server error: {e}")
            response_code = 500
            response = {"error": "Internal server error"}

        # Send response
        self.send_response(response_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        if response:
            response_json = json.dumps(response)
            self.wfile.write(response_json.encode('utf-8'))

    def log_message(self, format, *args):
        """Override to customize log format"""
        print(f"[{self.address_string()}] {format % args}")


def main():
    PORT = 8000
    HOST = "0.0.0.0"
    
    print("🧠 Initializing OCR Neural Network...")
    print("This may take a few minutes on first run as the network trains on initial data...")
    
    try:
        # Initialize the neural network
        global nn
        nn = OCRNeuralNetwork()
        print("✅ Neural network initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize neural network: {e}")
        print("Make sure you have numpy installed and the training data is available.")
        print("Server encountered a critical error and is shutting down.")
        sys.exit(1)

    # Start the server
    try:
        with socketserver.TCPServer((HOST, PORT), OCRHandler) as httpd:
            print(f"🚀 OCR Server running at:")
            print(f"   • Local: http://localhost:{PORT}")
            print(f"   • Network: http://{socketserver.socket.gethostbyname(socketserver.socket.gethostname())}:{PORT}")
            print("📝 Open your browser and navigate to the URL above")
            print("🛑 Press Ctrl+C to stop the server")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("Server encountered a critical error and is shutting down.")
        sys.exit(1)


if __name__ == "__main__":
    main()