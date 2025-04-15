from flask import Flask, render_template, request, jsonify, send_from_directory, abort
import os
import json
import argparse

app = Flask(__name__)

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Flask app for image viewer and click capture')
    parser.add_argument('--img_folder', type=str, required=False, default='static/images', help='Path to images folder')
    parser.add_argument('--metadata', type=str, default='rel_images.json', help='Path to metadata file')
    parser.add_argument('--click_file', type=str, default='clicks.json', help='Path to click data file')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    return parser

def load_metadata(metadata_file):
    if not os.path.exists(metadata_file):
        print(f"Metadata file {metadata_file} does not exist.")
        return {}
    
    with open(metadata_file, 'r') as f:
        return json.load(f)

def init_click_data(click_file):
    if not os.path.exists(click_file):
        with open(click_file, 'w') as f:
            json.dump([], f)

def get_last_labeled_image(click_file, image_list):
    if os.path.exists(click_file):
        with open(click_file, 'r') as f:
            click_data = json.load(f)
            if click_data:
                last_image = click_data[-1]['image']  # Get the last clicked image
                if last_image in image_list:
                    last_index = image_list.index(last_image)
                    return image_list[(last_index + 1) % len(image_list)]  # Get the next image
    return image_list[0]  # If no clicks, return the first image

@app.route('/')
def index():
    image_list = list(image_metadata.keys())  # Get all images
    current_image = get_last_labeled_image(CLICK_DATA_FILE, image_list)
    
    metadata = image_metadata[current_image]
    return render_template('index.html', image=current_image, metadata=metadata, images=image_list, all_metadata=image_metadata)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'static/images'), filename)

@app.route('/next_image/<current>', methods=['GET'])
def next_image(current):
    image_list = list(image_metadata.keys())
    current_index = image_list.index(current)
    next_index = (current_index + 1) % len(image_list)
    next_image = image_list[next_index]
    metadata = image_metadata[next_image]
    return jsonify({'image': next_image, 'metadata': metadata})

@app.route('/save_click', methods=['POST'])
def save_click():
    data = request.json
    image_name = data['image']
    x, y = data['x'], data['y']

    with open(CLICK_DATA_FILE, 'r') as f:
        click_data = json.load(f)

    click_data.append({'image': image_name, 'x': x, 'y': y})

    with open(CLICK_DATA_FILE, 'w') as f:
        json.dump(click_data, f)

    return jsonify({'status': 'success'})

@app.route('/reset_clicks/<image_name>', methods=['POST'])
def reset_clicks(image_name):
    """Reset click points for the specified image in clicks.json."""
    if not os.path.exists(CLICK_DATA_FILE):
        return jsonify({'status': 'error', 'message': 'clicks.json not found.'}), 404

    # Load the current click data
    with open(CLICK_DATA_FILE, 'r') as f:
        click_data = json.load(f)

    # Filter out clicks for the specified image
    filtered_clicks = [click for click in click_data if click['image'] != image_name]

    # Save the filtered clicks back to the file
    with open(CLICK_DATA_FILE, 'w') as f:
        json.dump(filtered_clicks, f)

    return jsonify({'status': 'success', 'message': f'Click points for {image_name} reset.'})

@app.route('/api/clicks', methods=['GET'])
def get_clicks():
    if not os.path.exists(CLICK_DATA_FILE):
        return jsonify({'status': 'error', 'message': 'clicks.json not found.'}), 404

    with open(CLICK_DATA_FILE, 'r') as f:
        click_data = json.load(f)

    return jsonify({'status': 'success', 'data': click_data})

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()

    IMAGE_FOLDER = args.img_folder  # Path to images folder
    METADATA_FILE = args.metadata
    CLICK_DATA_FILE = args.click_file

    image_metadata = load_metadata(METADATA_FILE)
    init_click_data(CLICK_DATA_FILE)

    app.run(debug=True, port=args.port)
