from flask import Flask, request, jsonify, render_template
from threading import Lock, Thread
from queue import Queue
import time
import os
import numpy as np
from scipy.optimize import least_squares

app = Flask(__name__)

anchor_data = {}
data_lock = Lock() 
trilateration_queue = Queue()
position_log = []


ANCHORS = {
    "7c:df:a1:0f:af:45": (-2.80, 0.00, 2.08),  # Anchor 1
    "7c:df:a1:0f:af:01": (0.00, 3.58, 2.04),  # Anchor 2
    "7c:df:a1:0e:d7:eb": (3.43, 0.00, 1.84),  # Anchor 3
}

THRESHOLD_SECONDS = 30

def perform_trilateration(distances, anchors):
    try:
        anchor_macs = ["7c:df:a1:0f:af:45", "7c:df:a1:0f:af:01", "7c:df:a1:0e:d7:eb"]
        anchor_positions = [np.array(anchors[mac]) for mac in anchor_macs]
        measured_distances = np.array(distances)
        
        if any(d <= 0 or d > 30 for d in measured_distances):
            print("Error: Invalid distances.")
            return None

        # Initial guess for Levenberg-Marquardt optimization
        initial_guess = np.mean(anchor_positions, axis=0)

        # Residual function for least squares
        def residual(position):
            errors = []
            for P, r in zip(anchor_positions, measured_distances):
                calculated_r = np.linalg.norm(position - P)
                errors.append(calculated_r - r)
            return errors
        
        result = least_squares(residual, initial_guess, method='lm', max_nfev=100)
        
        if not result.success:
            print(f"Optimization failed: {result.message}")
            return None

    
        x, y, z = result.x
        
  
        residuals = result.fun
        avg_error = np.mean(np.abs(residuals))
        print(f"Average residual error: {avg_error:.2f} meters")

        return {"x": float(x), "y": float(y), "z": float(z)}
    
    except Exception as e:
        print(f"Error in trilateration calculation: {e}")
        return None


def trilateration_worker():
    while True:
        macs, distances = trilateration_queue.get()
        print(f"Processing trilateration for MACs: {macs} with distances: {distances}")
        position = perform_trilateration(distances, ANCHORS)
        if position:
            print(f"Calculated position: {position}")
            with data_lock:
                position_log.append({"macs": macs, "distances": distances, "position": position, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')})
        else:
            print("Failed to calculate position")
        trilateration_queue.task_done()

Thread(target=trilateration_worker, daemon=True).start()

@app.route('/ftm_data', methods=['POST'])
def receive_ftm_data():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data received'}), 400

    anchor_mac, distance, timestamp = data.get('anchor_mac'), data.get('distance'), data.get('timestamp')
    with data_lock:
        if anchor_data and (timestamp - min(entry['timestamp'] for entry in anchor_data.values())) / 1000.0 > THRESHOLD_SECONDS:
            print("Threshold exceeded. Clearing stale data.")
            anchor_data.clear()
        anchor_data[anchor_mac] = {'distance': distance, 'timestamp': timestamp}

        if all(mac in anchor_data for mac in ANCHORS.keys()):
            distances = [anchor_data[mac]['distance'] for mac in ANCHORS.keys()]
            trilateration_queue.put((list(ANCHORS.keys()), distances))
            anchor_data.clear()
    
    print(f"Received data - Anchor MAC: {anchor_mac}, Distance: {distance}, Timestamp: {timestamp}")
    return jsonify({'status': 'success'}), 200

@app.route('/logs', methods=['GET'])
def get_logs():
    with data_lock:
        return jsonify(position_log)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/visualization')
def visualization():
    return render_template('index_viz.html')

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    from waitress import serve
    print("Starting server on port 130...")
    serve(app, host='0.0.0.0', port=130)