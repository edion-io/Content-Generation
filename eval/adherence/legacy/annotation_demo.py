"""
Dataset Annotation Tool

This script provides a Gradio-based web interface for annotating datasets.
It allows users to load samples, edit parameters, generate new instructions,
and save annotations with adherence scores. Resuming is supported if both 
--input-data and --annotated-data are provided.

Usage:
    **Install dependencies:**
    `pip install gradio pandas pyspellchecker inflect rapidfuzz matplotlib`

    **Run the script:**
    ```bash
    python annotation_demo.py \
        --input-data /home/xxx/xxx/datasets/data/complete_data_subset-60.csv \
        --annotated-data /home/xxx/xxx/Content-Generation/eval/adherence/annotated_dataset.csv
    ```
    replace the paths with your own absolute paths.

    If you got an error like this: `AttributeError: 'int' object has no attribute 'expandtabs'`,
    you need to check the data path again and make sure they are correct.

    **Open your browser and visit url:**
    `http://0.0.0.0:7860`
"""
import gradio as gr
import pandas as pd
import os
import sys
from datetime import datetime
import re
import argparse
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import functions from preprocess.py
from eval.adherence.legacy.preprocess import instructionize

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Dataset Annotation Tool")
parser.add_argument("--input-data", help="Path to the input CSV file.")
parser.add_argument("--annotated-data", default=os.path.join(project_root, "eval", "adherence", "annotated_dataset.csv"), help="Path to save/load annotated data.")
args = parser.parse_args()

# Load the dataset
def load_data():
    try:
        df = pd.read_csv(args.input_data)
        # Add ID column if not present
        df.reset_index(drop=True, inplace=True)
        df['ID'] = df.index
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_existing_annotations():
    """Load existing annotations from the saved CSV file"""
    global annotated_data, annotated_ids
    
    try:
        if os.path.exists(args.annotated_data):
            annotated_df = pd.read_csv(args.annotated_data)
            print(f"📄 Found existing annotations file with {len(annotated_df)} records: {args.annotated_data}")
            
            # Convert dataframe back to list of dictionaries
            annotated_data = annotated_df.to_dict('records')
            
            # Populate the annotated_ids set
            annotated_ids = set(annotated_df['original_id'].tolist())
            
            print(f"✅ Loaded {len(annotated_data)} previous annotations")
            print(f"📝 Annotated IDs: {sorted(list(annotated_ids))}")
            return True
        else:
            print(f"📄 No existing annotations file found ('{args.annotated_data}'). Starting fresh.")
            annotated_data = []
            annotated_ids = set()
            return False
    except Exception as e:
        print(f"⚠️  Error loading existing annotations: {e}")
        print("📄 Starting with empty annotations.")
        annotated_data = []
        annotated_ids = set()
        return False

def convert_latex_to_html(text):
    """Convert LaTeX content to HTML with MathJax support"""
    if not text:
        return ""
    
    # Convert LaTeX equations to MathJax format
    # Handle display equations \[ ... \]
    text = re.sub(r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r'$$\1$$', text, flags=re.DOTALL)
    
    # Handle inline math \( ... \)
    text = re.sub(r'\\begin\{align\*\}(.*?)\\end\{align\*\}', r'$$\\begin{align}\1\\end{align}$$', text, flags=re.DOTALL)
    
    # Convert enumerate environments to HTML lists
    text = re.sub(r'\\begin\{enumerate\}', '<ol>', text)
    text = re.sub(r'\\end\{enumerate\}', '</ol>', text)
    text = re.sub(r'\\begin\{enumerate\}\[label=\\alph\*\.\]', '<ol type="a">', text)
    text = re.sub(r'\\begin\{enumerate\}\[label=\\Roman\*\.\]', '<ol type="I">', text)
    
    # Convert itemize to unordered lists
    text = re.sub(r'\\begin\{itemize\}', '<ul>', text)
    text = re.sub(r'\\end\{itemize\}', '</ul>', text)
    
    # Convert \item to <li>
    text = re.sub(r'\\item\s+', '<li>', text)
    
    # Convert basic LaTeX formatting
    text = re.sub(r'\\textbf\{([^}]+)\}', r'<strong>\1</strong>', text)
    text = re.sub(r'\\textit\{([^}]+)\}', r'<em>\1</em>', text)
    text = re.sub(r'\\subsection\*\{([^}]+)\}', r'<h4>\1</h4>', text)
    text = re.sub(r'\\paragraph\{([^}]+)\}', r'<h5>\1</h5>', text)
    
    # Handle tables (basic conversion)
    text = re.sub(r'\\begin\{tabular\}.*?\{.*?\}', '<table border="1" style="border-collapse: collapse;">', text)
    text = re.sub(r'\\end\{tabular\}', '</table>', text)
    text = re.sub(r'\\hline', '', text)
    text = re.sub(r'&', '</td><td>', text)
    text = re.sub(r'\\\\', '</td></tr><tr><td>', text)
    
    # Convert line breaks
    text = text.replace('\\\\', '<br>')
    text = text.replace('\n\n', '<br><br>')
    
    # Handle special LaTeX commands that should be preserved for MathJax
    # Keep math expressions intact
    
    # Wrap in HTML with MathJax
    html_content = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; padding: 10px;">
        {text}
    </div>
    <script>
        if (typeof MathJax !== 'undefined') {{
            MathJax.typesetPromise();
        }}
    </script>
    """
    
    return html_content

# Global variables
df = load_data()
current_sample = {}
# Initialize annotation tracking variables
load_existing_annotations()  # This will populate annotated_data and annotated_ids

def get_sample_list():
    """Get list of sample IDs for selection with annotation status"""
    if df.empty:
        return []
    sample_list = []
    for i, row in df.iterrows():
        # Mark annotated samples with ✅
        status_mark = "✅ " if i in annotated_ids else ""
        sample_list.append(f"{status_mark}ID {i}: {row['subject']} - {row['exercise_type'][:50]}...")
    return sample_list

def load_sample(selected_id):
    """Load a sample based on selected ID"""
    if not selected_id:
        return "", "", "", "", "", "", "", "", "", "", "", "**No sample selected**", 0, 0, 0, 0, 0
    
    # Extract ID from the selection string (handle both marked and unmarked)
    try:
        # Remove the ✅ mark if present
        clean_id = selected_id.replace("✅ ", "")
        id_num = int(clean_id.split(':')[0].replace('ID ', ''))
        row = df.iloc[id_num]
        
        global current_sample
        current_sample = {
            'id': id_num,
            'instruction': row['instruction'],
            'subject': row['subject'],
            'exercise_type': row['exercise_type'],
            'difficulty': row['difficulty'],
            'grade_level': str(row['grade_level']),
            'modifier': row['modifier'],
            'generation': row['generation']
        }
        
        # Check if this sample has previous annotations
        previous_annotation = None
        for annotation in annotated_data:
            if annotation['original_id'] == id_num:
                previous_annotation = annotation
                break
        
        # Prepare content for display with LaTeX rendering
        orig_instruction_md = f"**Instruction:**\n\n{row['instruction']}"
        orig_generation_html = convert_latex_to_html(row['generation'])
        
        # Initialize adherence scores
        adherence_scores = [0, 0, 0, 0, 0]  # Default values
        
        # If there's a previous annotation, use the edited values
        if previous_annotation:
            edit_subject = previous_annotation['new_subject']
            edit_exercise_type = previous_annotation['new_exercise_type']
            edit_difficulty = previous_annotation['new_difficulty']
            edit_grade_level = str(previous_annotation['new_grade_level'])
            edit_modifier = previous_annotation['new_modifier']
            edit_instruction_md = f"**Updated Instruction:\n\n** {previous_annotation['new_instruction']}"
            
            # Load previous adherence scores
            adherence_scores = [
                int(previous_annotation.get('adherence_subject', 0)),
                int(previous_annotation.get('adherence_exercise_type', 0)),
                int(previous_annotation.get('adherence_difficulty', 0)),
                int(previous_annotation.get('adherence_grade_level', 0)),
                int(previous_annotation.get('adherence_modifier', 0))
            ]
            
            # Update current_sample with previous edits
            current_sample.update({
                'new_instruction': previous_annotation['new_instruction'],
                'new_subject': edit_subject,
                'new_exercise_type': edit_exercise_type,
                'new_difficulty': edit_difficulty,
                'new_grade_level': edit_grade_level,
                'new_modifier': edit_modifier
            })
        else:
            # Use original values if no previous annotation
            edit_subject = row['subject']
            edit_exercise_type = row['exercise_type']
            edit_difficulty = row['difficulty']
            edit_grade_level = str(row['grade_level'])
            edit_modifier = row['modifier']
            edit_instruction_md = f"**Instruction:** {row['instruction']}"
        
        # Return values for left panel (original data), right panel (editable), and adherence scores
        return (
            row['subject'],
            row['exercise_type'], 
            row['difficulty'],
            str(row['grade_level']),
            row['modifier'],
            orig_instruction_md,
            orig_generation_html,
            # Return values for right panel (editable)
            edit_subject,
            edit_exercise_type,
            edit_difficulty, 
            edit_grade_level,
            edit_modifier,
            edit_instruction_md,
            # Return adherence scores
            adherence_scores[0],  # subject
            adherence_scores[1],  # exercise
            adherence_scores[2],  # difficulty
            adherence_scores[3],  # grade
            adherence_scores[4]   # modifier
        )
    except Exception as e:
        print(f"Error loading sample: {e}")
        return "", "", "", "", "", "**No data available**", "", "", "", "", "", "**No data available**", 0, 0, 0, 0, 0

def update_instruction(subject, exercise_type, difficulty, grade_level, modifier):
    """Generate new instruction based on edited parameters using the instructionize function"""
    if not all([subject, exercise_type, difficulty, grade_level]):
        return "Please fill in all required fields."
    
    try:
        # Create a header string in the format expected by instructionize
        # Format: (Subject) (Exercise_Type) D Grade (Modifier)
        # Handle cases where parameters might be default values
        subject_param = subject if subject and subject != 'S' else 'S'
        exercise_param = exercise_type if exercise_type and exercise_type != 'T' else 'T'
        grade_param = grade_level if grade_level and grade_level != 'G' else 'G'
        modifier_param = modifier if modifier and modifier != 'M' else 'M'
        
        # Construct the header string
        header = f"({subject_param}) ({exercise_param}) D {grade_param} ({modifier_param})"
        
        # Use the instructionize function from preprocess.py
        new_instruction = instructionize(header)
        
        # Update current sample
        global current_sample
        if current_sample:
            current_sample.update({
                'new_instruction': new_instruction,
                'new_subject': subject,
                'new_exercise_type': exercise_type,
                'new_difficulty': difficulty,
                'new_grade_level': grade_level,
                'new_modifier': modifier
            })
        
        updated_instruction_md = f"**Updated Instruction:** {new_instruction}"
        
        return updated_instruction_md
        
    except Exception as e:
        print(f"Error generating instruction: {e}")
        return f"Error generating instruction: {str(e)}"

def save_annotation(adherence_subject, adherence_exercise, adherence_difficulty, 
                   adherence_grade, adherence_modifier):
    """Save the annotated data with adherence scores"""
    global current_sample, annotated_data, annotated_ids
    
    if not current_sample:
        return "No sample loaded. Please select a sample first.", "", gr.Dropdown(choices=get_sample_list())
    
    # Validate adherence scores with new constraints
    try:
        # Subject adherence: 0 or 1 only
        if adherence_subject not in [0, 1]:
            return "Subject adherence must be 0 or 1.", "", gr.Dropdown(choices=get_sample_list())
        
        # Other adherence scores: 0-5
        other_scores = [adherence_exercise, adherence_difficulty, adherence_grade, adherence_modifier]
        for score in other_scores:
            if score < 0 or score > 5:
                return "Exercise, Difficulty, Grade, and Modifier adherence must be between 0-5.", "", gr.Dropdown(choices=get_sample_list())
        
        adherence_scores = [
            int(adherence_subject),  # 0 or 1
            float(adherence_exercise),  # 0-5
            float(adherence_difficulty),  # 0-5
            float(adherence_grade),  # 0-5
            float(adherence_modifier)  # 0-5
        ]
    except (ValueError, TypeError):
        return "Invalid adherence scores. Subject: 0 or 1, Others: 0-5.", "", gr.Dropdown(choices=get_sample_list())
    
    # Create annotated record
    annotated_record = {
        'original_id': current_sample['id'],
        'original_instruction': current_sample['instruction'],
        'original_subject': current_sample['subject'],
        'original_exercise_type': current_sample['exercise_type'],
        'original_difficulty': current_sample['difficulty'],
        'original_grade_level': current_sample['grade_level'],
        'original_modifier': current_sample['modifier'],
        'original_generation': current_sample['generation'],
        'new_instruction': current_sample.get('new_instruction', current_sample['instruction']),
        'new_subject': current_sample.get('new_subject', current_sample['subject']),
        'new_exercise_type': current_sample.get('new_exercise_type', current_sample['exercise_type']),
        'new_difficulty': current_sample.get('new_difficulty', current_sample['difficulty']),
        'new_grade_level': current_sample.get('new_grade_level', current_sample['grade_level']),
        'new_modifier': current_sample.get('new_modifier', current_sample['modifier']),
        'adherence_subject': adherence_scores[0],
        'adherence_exercise_type': adherence_scores[1],
        'adherence_difficulty': adherence_scores[2],
        'adherence_grade_level': adherence_scores[3],
        'adherence_modifier': adherence_scores[4],
        'annotation_timestamp': datetime.now().isoformat()
    }
    
    # Check if this ID was already annotated and update if so
    existing_index = None
    for i, record in enumerate(annotated_data):
        if record['original_id'] == current_sample['id']:
            existing_index = i
            break
    
    if existing_index is not None:
        annotated_data[existing_index] = annotated_record
        action = "updated"
    else:
        annotated_data.append(annotated_record)
        action = "saved"
    
    # Mark this ID as annotated
    annotated_ids.add(current_sample['id'])
    
    # Save to file
    annotated_df = pd.DataFrame(annotated_data)
    output_file = args.annotated_data
    annotated_df.to_csv(output_file, index=False)
    
    # Update the dropdown with new annotation status
    updated_choices = get_sample_list()

    # Find the current sample in the updated choices (it should now have the ✅ mark)
    current_sample_with_mark = None
    for choice in updated_choices:
        if f"ID {current_sample['id']}:" in choice:
            current_sample_with_mark = choice
            break

    return (
        f"✅ Annotation {action}! Total annotations: {len(annotated_data)}", 
        f"Saved to: {output_file}",
        gr.Dropdown(choices=updated_choices, value=current_sample_with_mark)
    )

def next_sample(current_dropdown_value):
    """Move to the next sample in the dataset"""
    global df
    
    if df.empty:
        return gr.Dropdown(value=None)
    
    # Get the current sample list (this will have updated annotation marks)
    sample_list = get_sample_list()
    
    if not sample_list:
        return gr.Dropdown(value=None)
    
    # If no current sample is selected, start with the first one
    if not current_dropdown_value:
        return gr.Dropdown(choices=sample_list, value=sample_list[0])
    
    try:
        # Find current index in the dropdown list
        current_index = sample_list.index(current_dropdown_value)
        
        # Move to next sample (with wraparound to beginning)
        next_index = (current_index + 1) % len(sample_list)
        next_sample_value = sample_list[next_index]
        
        return gr.Dropdown(choices=sample_list, value=next_sample_value)
        
    except (ValueError, IndexError):
        # If current sample not found, start from the beginning
        return gr.Dropdown(choices=sample_list, value=sample_list[0])

def get_initial_sample():
    """Get the initial sample to display when the app starts"""
    sample_list = get_sample_list()
    if sample_list:
        return sample_list[0]
    return None

# Create Gradio interface
with gr.Blocks(title="Dataset Annotation Tool", theme=gr.themes.Soft(), css="""
    /* Apply uniform font family throughout the interface */
    * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
    }
    
    /* Ensure consistent font weight and size */
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
    }
    
    /* Style for text inputs and textareas */
    input, textarea, select {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
    }
    
    /* Style for labels and markdown */
    label, .markdown {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
    }
    
    /* Style for dropdowns */
    .dropdown {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
    }
""") as demo:
    # Add MathJax support
    gr.HTML("""
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: true,
            processEnvironments: true
        },
        options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        }
    };
    </script>
    """)
    
    gr.Markdown("# 📊 Dataset Annotation Tool")
    gr.Markdown("Select a sample from the list below to start annotation. ✅ indicates already annotated samples.")
    gr.Markdown("**Resume Feature:** Previously saved annotations are automatically loaded. Annotated samples show their previous values and scores.")
    
    # Get initial sample data
    sample_choices = get_sample_list()
    initial_sample = sample_choices[0] if sample_choices else None
    initial_data = load_sample(initial_sample) if initial_sample else ("", "", "", "", "", "", "", "", "", "", "", "**No sample selected**", 0, 0, 0, 0, 0)
    
    with gr.Row():
        # Sample selection
        sample_dropdown = gr.Dropdown(
            choices=sample_choices,
            label="📋 Select Sample",
            info="Choose a sample to annotate (✅ = already annotated)",
            interactive=True,
            value=initial_sample
        )

    # Workspace area
    with gr.Row():
        # Left panel - Original data (read-only)
        with gr.Column(scale=1):
            gr.Markdown("### 📖 Original Data")
            with gr.Group():
                orig_subject = gr.Textbox(label="Subject", interactive=False, value=initial_data[0])
                orig_exercise_type = gr.Textbox(label="Exercise Type", interactive=False, value=initial_data[1])
                orig_difficulty = gr.Textbox(label="Difficulty", interactive=False, value=initial_data[2])
                orig_grade_level = gr.Textbox(label="Grade Level", interactive=False, value=initial_data[3])
                orig_modifier = gr.Textbox(label="Modifier", interactive=False, value=initial_data[4])
            
            orig_instruction = gr.Markdown(label="Instruction", value=initial_data[5])
            gr.Markdown("**Generation:**")
            orig_generation = gr.HTML(label="Generation", value=initial_data[6])
        
        # Right panel - Editable data
        with gr.Column(scale=1):
            gr.Markdown("### ✏️ Editable Data")
            with gr.Group():
                edit_subject = gr.Textbox(label="Subject", interactive=True, value=initial_data[7])
                edit_exercise_type = gr.Textbox(label="Exercise Type", interactive=True, value=initial_data[8])
                edit_difficulty = gr.Textbox(label="Difficulty", interactive=True, value=initial_data[9])
                edit_grade_level = gr.Textbox(label="Grade Level", interactive=True, value=initial_data[10])
                edit_modifier = gr.Textbox(label="Modifier", interactive=True, value=initial_data[11])
                
                update_btn = gr.Button("🔄 Update Instruction", variant="primary")
            
            edit_instruction = gr.Markdown(label="Updated Instruction", value=initial_data[12])
            gr.Markdown("**Note:** The generation content is displayed in the left panel for reference.")
    
    # Bottom panel - Adherence scoring
    gr.Markdown("### 🎯 Adherence Scoring")
    gr.Markdown("**Subject Adherence:** 0 (No) or 1 (Yes) | **Other Adherence:** 0-5 scale (0=Poor, 5=Excellent)")
    
    with gr.Row():
        adherence_subject = gr.Number(
            label="Subject Adherence (0 or 1)", 
            value=initial_data[13], 
            minimum=0, 
            maximum=1, 
            step=1,
            info="0 = Incorrect subject, 1 = Correct subject"
        )
        adherence_exercise = gr.Number(
            label="Exercise Type Adherence (0-4)", 
            value=initial_data[14], 
            minimum=0, 
            maximum=4, 
            step=1,
            info="0 = Poor match, 4 = Perfect match"
        )
        adherence_difficulty = gr.Number(
            label="Difficulty Adherence (0-4)", 
            value=initial_data[15], 
            minimum=0, 
            maximum=4, 
            step=1,
            info="0 = Wrong difficulty, 4 = Perfect difficulty"
        )
        adherence_grade = gr.Number(
            label="Grade Level Adherence (0-4)", 
            value=initial_data[16], 
            minimum=0, 
            maximum=4, 
            step=1,
            info="0 = Wrong grade, 4 = Perfect grade level"
        )
        adherence_modifier = gr.Number(
            label="Modifier Adherence (0-4)", 
            value=initial_data[17], 
            minimum=0, 
            maximum=4, 
            step=1,
            info="0 = Ignores modifier, 4 = Follows modifier perfectly"
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            next_btn = gr.Button("➡️ Next Sample", variant="secondary", size="lg")
            save_btn = gr.Button("💾 Save Annotation", variant="primary", size="lg")
        status_msg = gr.Textbox(label="Status", interactive=False)
        file_info = gr.Textbox(label="File Info", interactive=False)
    
    
    # Event handlers
    sample_dropdown.change(
        fn=load_sample,
        inputs=[sample_dropdown],
        outputs=[orig_subject, orig_exercise_type, orig_difficulty, orig_grade_level, 
                orig_modifier, orig_instruction, orig_generation, edit_subject, edit_exercise_type, 
                edit_difficulty, edit_grade_level, edit_modifier, edit_instruction,
                adherence_subject, adherence_exercise, adherence_difficulty, adherence_grade, adherence_modifier]
    )
    
    update_btn.click(
        fn=update_instruction,
        inputs=[edit_subject, edit_exercise_type, edit_difficulty, edit_grade_level, edit_modifier],
        outputs=[edit_instruction]
    )
    
    save_btn.click(
        fn=save_annotation,
        inputs=[adherence_subject, adherence_exercise, adherence_difficulty, 
               adherence_grade, adherence_modifier],
        outputs=[status_msg, file_info, sample_dropdown]
    )

    next_btn.click(
        fn=next_sample,
        inputs=[sample_dropdown],
        outputs=[sample_dropdown]
    )

if __name__ == "__main__":
    # Check if data file exists
    if not os.path.exists(args.input_data):
        print(f"⚠️  Data file not found. Please ensure '{args.input_data}' exists.")
    else:
        print(f"📊 Loaded {len(df)} samples from dataset: {args.input_data}")
        if len(annotated_data) > 0:
            print(f"🎯 Resumed with {len(annotated_data)} existing annotations")
            print(f"📝 Progress: {len(annotated_ids)}/{len(df)} samples annotated ({len(annotated_ids)/len(df)*100:.1f}%)")
        else:
            print(f"🆕 Starting fresh annotation session. Annotations will be saved to '{args.annotated_data}'")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 