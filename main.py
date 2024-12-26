from flask import Flask, render_template, render_template_string, request, send_from_directory
import requests
import os
import cv2

app = Flask(__name__)
#app = Flask(__name__, template_folder='path_to_templates')



UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


ROBOFLOW_API_KEY = "qvg1QVjLoSU8vPYnhewB"
#"https://detect.roboflow.com/fencing-en-garde/5?api_key=qvg1QVjLoSU8vPYnhewB"
#ROBOFLOW_MODEL_URL = "https://detect.roboflow.com/fencing-en-garde/5?api_key=qvg1QVjLoSU8vPYnhewB"
#https://detect.roboflow.com/[model-name]/[version]?api_key=[your-api-key]

ROBOFLOW_MODEL_URL = "https://detect.roboflow.com/fencing-en-garde"
#"https://detect.roboflow.com/fencing-en-garde/5"

ROBOFLOW_VERSION = "5"  

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']





@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part in the request", 400

    file = request.files['image']

    if file.filename == '':
        return "No file selected", 400
    if not file.filename:
        return "Invalid file", 400
    
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Send image to Roboflow API for processing
        with open(filepath, 'rb') as img:
            response = requests.post(
                f"{ROBOFLOW_MODEL_URL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}",
                files={"file": img}
            )
            # Check if the request was successful (status code 200)
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")
            print(f"Filepath: {filepath}")
            print(f"File exists: {os.path.exists(filepath)}")

            # Only attempt to parse JSON if the status code is 200 (OK)
            if response.status_code == 200:
                try:
                    predictions = response.json().get("predictions", [])
                except requests.exceptions.JSONDecodeError:
                    return f"Error: Unable to parse JSON. Response: {response.text}", 500
            else:
                return f"Error in Roboflow API: {response.text}", response.status_code

            #if response.status_code == 200:
                #predictions = response.json() 
                #return predictions
            #else:
                #return f"Error in Roboflow API: {response.text}", response.status_code
        #return f"Error in Roboflow API: {response.text}", 500
    if not allowed_file(file.filename):
        extension = file.filename.rsplit('.', 1)[1] if '.' in file.filename else "none"
        return f"Invalid file type: .{extension}. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", 400



    # List of colors to cycle through
    colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255)  # Cyan
    ]

    class_colors = {
        "Arm down": (0, 255, 0),     # Green
        "Arm good distance": (255, 0, 0),  # Blue
        "Arm too far forward": (0, 0, 255), # Red
        "Arm up": (255, 255, 0),       # Yellow
        "Centred body": (255, 0, 255),  # Magenta
        "Leaning forward": (0, 255, 255),  # Cyan
        "Legs bent perfect": (128, 128, 0),#Olive
        "Legs bent too little": (128, 0, 128),# Purple
        "Legs bent too much": (0, 128, 128),#Teal
        "Long stance": (128, 0, 0),    # Maroon
        "Medium stance": (0, 128, 0),# Dark Green
        "Perfect arm angle": (0, 0, 128), # Navy
        "Short stance": (128, 128, 128) # Gray
    }
    
    # Mutually exclusive groups of classes
    exclusive_groups = {
        "arm_position": ["Arm down", "Arm good distance", "Arm too far forward", "Arm up", "Perfect arm angle"],
        "body_position": ["Centred body", "Leaning forward"],
        "legs_position": ["Legs bent perfect", "Legs bent too little", "Legs bent too much"],
        "stance": ["Long stance", "Medium stance", "Short stance"]
    }

    # Function to filter predictions based on the highest confidence in each group
    def filter_predictions(predictions, groups):
        filtered_predictions = []
        for group_name, classes in groups.items():
            group_predictions = [pred for pred in predictions if pred["class"] in classes]
            if group_predictions:
                # Select prediction with highest confidence
                best_prediction = max(group_predictions, key=lambda x: x["confidence"])
                filtered_predictions.append(best_prediction)
        return filtered_predictions
        
    # Visualize predictions on the image
    img = cv2.imread(filepath)

    # Filter predictions before visualization
    #predictions = predictions["predictions"]  # Extract actual prediction data from Roboflow response
    filtered_predictions = filter_predictions(predictions, exclusive_groups)


    for idx, pred in enumerate(predictions):
        x, y, width, height = pred["x"], pred["y"], pred["width"], pred["height"]
        label = pred["class"]
        confidence = pred["confidence"]


        start_x = int(x - width / 2)
        start_y = int(y - height / 2)
        end_x = int(x + width / 2)
        end_y = int(y + height / 2)

 
        color = colors[idx % len(colors)] 

        # Draw bounding box and label
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, 2)
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(img, label_text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



    # Save visualized image
    visualized_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"visualized_{file.filename}")
    cv2.imwrite(visualized_filepath, img)


    labeled_classes = {pred["class"] for pred in filtered_predictions}


    print("Classes labeled on the image:", labeled_classes)

 
    class_list_html = "<br>".join(labeled_classes)

    # Define positive and negative attributes
    positive_classes = {"Arm good distance", "Centred body", "Legs bent perfect", "Medium stance", "Perfect arm angle"}
    negative_classes = {"Arm down", "Arm too far forward", "Arm up", "Leaning forward", "Legs bent too little", "Legs bent too much", "Long stance", "Short stance"}


    # messages for each class
    class_messages = {
        "Arm down": "Your arm is pointed downwards, meaning it will take longer to hit your opponent. Try raising it.",
        "Arm good distance": "Your arm is at a good distance. Well done!",
        "Arm too far forward": "Your arm is too far forward, meaning your less prepared to parry or defend yourself. Try pulling it back slightly.",
        "Arm up": "Your arm is up, meaning that it will take longer to hit your opponent. Lower it to a neutral position.",
        "Centred body": "Your body is centered so you will be fast changing direction on your feet. Great job!",
        "Leaning forward": "You are leaning forward- whilst you may feel like you have an extra reach, this makes it harder for you to change direction on your feet. Try using your core to straighten your back up.",
        "Legs bent perfect": "Your legs are bent perfectly so you will be both ready on your feet and ready to lunge. Keep it up!",
        "Legs bent too little": "Your legs are not bent enough so they won't be loaded ready to lunge or attack. Try bending them a bit more.",
        "Legs bent too much": "Your legs are bent too much making it harder to move on your feet. Try straightening them slightly.",
        "Long stance": "Your feet are too far apart making you slower on your feet. Try moving your feet slightly together.",
        "Medium stance": "Your feet are a good distance apart so you will be balanced and able to move quickly on your feet. Looking good!",
        "Perfect arm angle": "Your arm is neither too far up or too far down so your point will take less time to hit your opponent. Perfect!",
        "Short stance": "Your feet are too close together so you will be less balanced and fast on your feet. Try moving your feet further apart."
    }
    # categorise feedback as positive/negative
    positive_feedback = []
    negative_feedback = []

    for pred in filtered_predictions:
        class_name = pred["class"]
        message = class_messages.get(class_name, "No specific message for this class.")
        if class_name in positive_classes:
            positive_feedback.append(message)
        elif class_name in negative_classes:
            negative_feedback.append(message)

    
    positive_feedback_html = "<br>".join(positive_feedback) or "No positive feedback detected."
    negative_feedback_html = "<br>".join(negative_feedback) or "No areas for improvement detected."

    # Generate messages for recognized classes
    messages = [class_messages.get(pred["class"], "No specific message for this class.") for pred in filtered_predictions]

    
    messages_html = "<br>".join(messages)

    
    return f"""
    <html><head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>En garde!</title>
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <!-- Custom CSS (optional) -->
  <style>
    body {{
      background-color: #f8f9fa;
    }}
      /* Make the header (toolbar) red */
      header {{
        background-color: #901B02; /* Red color */
      }}

      /* Optional: Change the text color of the header (toolbar) links to white for contrast */
      header .nav-link {{
        color: white; /* White text for links */
      }}

      /* Optional: Hover effect for links */
      header .nav-link:hover {{
        color: #f8f9fa; /* Change to light color on hover */
      }}
      .custom-table {{
        background-color: #f8f9fa; /* Set table background color same as body */
        color: #000; /* Black text */
        border-color: #000; /* Black borders */
      }}

      .custom-table th {{
        background-color: #f8f9fa; /* Set header background color to match body */
        color: #000; /* Black text for header */
      }}

      .custom-table td {{
        background-color: #f8f9fa; /* Set table cell background to match the body */
        color: #000; /* Black text for table cells */
      }}

      .custom-table, .custom-table th, .custom-table td {{
        border: 1px solid #000; /* Set border color for table, header, and cells */
      }}
    </style>
  </style>
</head>
<body>
        <div class="d-flex h-100 text-center text-bg-dark">
          <div class="cover-container d-flex w-100 h-100 p-3 mx-auto flex-column">
            <header class="mb-auto">
              <div>
                <h3 class="float-md-start mb-0">En Garde!</h3>
                <nav class="nav nav-masthead justify-content-center float-md-end">
                  <a class="nav-link active" href="#">Home</a>
                  <a class="nav-link" href="#">Try it out!</a>
                  <a class="nav-link" href="#">Feedback</a>
                </nav>
              </div>
                </header>
        <h1>Predictions</h1>
        <img src="/uploads/visualized_{file.filename}" style="max-width: 100%; height: auto;">
<div class="table-container">

        <h2>Classes Labeled:</h2>
        <table class="table table-bordered custom-table">
          <thead>
            <tr>
              <th>Things done well</th>
              <th>Things to work on</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{positive_feedback_html}</td>
              <td>{negative_feedback_html}</td>
            </tr>
          </tbody>
        </table>
      </div>
        </body>
        </html>
        """
    
    
    # Return visualized image in the response



    #else:
            #return f"Error in Roboflow API: {response.text}", response.status_code


    
            
        
        





    
    #return "Invalid file type", 400



feedback_folder = 'feedback'
if not os.path.exists('feedback'):
    os.makedirs('feedback')

#@app.route('/')
#def feedback_form():
  #  return render_template_string(open('feedback').read())

@app.route('/feedback')
def feedback_form():
    return '''
    <!DOCTYPE html>
    <html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>En garde!</title>
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <!-- Custom CSS (optional) -->
  <style>
   
    body {
      background-color: #f8f9fa; 
      height: 100vh; 
      margin: 0; 
    }

 
    header {
      background-color: #901B02; 
    }

    /* Change the text color of the header (toolbar) links to white for contrast */
    header .nav-link {
      color: white; /* White text for links */
    }

    /* Hover effect for links */
    header .nav-link:hover {
      color: #f8f9fa; 
    }

    /* Center the title */
    h1 {
      text-align: center;
    }

    /* Ensure that the content is centered vertically */
    .cover-container {
      display: flex;
      flex-direction: column;
      justify-content: center;
      height: 100%;
    }

    .main-content {
      text-align: center; 
    }
      
  </style>
</head>
<body>
    <div class="d-flex h-100 text-center text-bg-dark">
      <div class="cover-container d-flex w-100 h-100 p-3 mx-auto flex-column">
        <header class="mb-auto">
          <div>
            <h3 class="float-md-start mb-0" style="text-align: center;">En Garde!</h3>
            <nav class="nav nav-masthead justify-content-center float-md-end">
              <a class="nav-link active" href="/">Home</a>
              <a class="nav-link" href="engarde">Try it out!</a>
              <a class="nav-link" href="feedback">Feedback</a>
            </nav>
          </div>
        </header>
    <body>
      <h1>Feedback Form</h1>
      <form action="/submit_feedback" method="POST">
        <label for="feedback">Your Feedback:</label><br><br>
        <textarea id="feedback" name="feedback" rows="4" cols="50" required></textarea><br><br>
        <button type="submit">Submit Feedback</button>
      </form>
      <!-- Bootstrap JS -->
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    '''

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form.get('feedback')  # Get feedback from form

    if feedback:
        # save feedback to text file
        with open(f"feedback/feedback_{len(os.listdir('feedback')) + 1}.txt", 'w') as f:
            f.write(feedback)
        return "<h2>Thank you for your feedback!</h2><a href='/'>Go back home</a>"
    return "<h2>Error: Feedback could not be saved. Please try again.</h2><a href='/'>Go back to home</a>"






@app.route('/upload', methods=['GET', 'POST'])
def upload_form():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Upload Image</title>
        </head>
        <body>
            <h1>Upload an Image</h1>
            <form action="/upload" method="POST" enctype="multipart/form-data">
                <label for="image">Choose an image:</label>
                <input type="file" id="image" name="image" accept="image/*">
                <br><br>
                <button type="submit">Upload</button>
            </form>
        </body>
        </html>
    ''')


@app.route('/')
def home():
    return render_template("home.html")


@app.route("/engarde")
def engarde():
    return render_template("engarde.html")


@app.route("/lunge")
def lunge():
    return render_template("lunge.html")
    
def index():
    return render_template("upload.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
