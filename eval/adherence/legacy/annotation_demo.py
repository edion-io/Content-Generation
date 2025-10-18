"""
Dataset Annotation Tool

This script provides a Gradio-based web interface for annotating datasets.
It allows users to load samples, edit parameters, generate new instructions,
and save annotations with adherence scores.

FLEXIBLE USAGE MODES:

1. **Annotation with Input Data (Recommended):**
   ```bash
   python annotation_demo.py \
       --input-data /path/to/original/data.csv \
       --annotated-data /path/to/annotations/annotated_dataset.csv
   ```
   - Loads all samples from input data
   - Resumes previous annotations if output file exists
   - Saves ALL samples (annotated + unannotated) to output file
   - Next time, can load output file directly for review

2. **Review Existing Annotations (Standalone):**
   ```bash
   python annotation_demo.py \
       --annotated-data /path/to/annotations/annotated_dataset.csv
   ```
   - Reviews and edits existing annotations without needing input data
   - All samples are already in the annotation file

**Key Feature:** When saving annotations, ALL samples (both annotated and unannotated)
are saved to the output file. Unannotated samples have empty/default values, allowing
you to resume annotation later without needing the original input data.

Installation:
    `pip install gradio pandas pyspellchecker inflect rapidfuzz matplotlib`

Open your browser and visit: `http://0.0.0.0:7860`
"""
import gradio as gr
import pandas as pd
import os
import sys
from datetime import datetime
import re
import argparse
from pathlib import Path
from typing import Optional

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import functions from preprocess.py
from eval.adherence.legacy.preprocess import instructionize

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Dataset Annotation Tool - Uniform Mode")
parser.add_argument("--input-data", default=None, help="Path to the input CSV file (optional).")
parser.add_argument("--annotated-data", default=os.path.join(project_root, "eval", "adherence", "annotated_dataset.csv"), help="Path to save/load annotated data.")
args = parser.parse_args()

# Determine mode based on available inputs
if args.input_data and os.path.exists(args.input_data):
    MODE = "ANNOTATION"  # Has input data to annotate
elif os.path.exists(args.annotated_data):
    MODE = "REVIEW"      # Only has annotated data
else:
    MODE = "ANNOTATION"  # Default to annotation mode (will error if no data found)

# Load the dataset
def load_data():
    """Load original dataset (if available)"""
    if args.input_data is None:
        return pd.DataFrame()
    
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
            # Normalize missing values so downstream code gets empty strings instead of NaN
            annotated_df.fillna('', inplace=True)
            print(f"📄 Found existing annotations file with {len(annotated_df)} records: {args.annotated_data}")
            
            # Convert dataframe back to list of dictionaries
            annotated_data = annotated_df.to_dict('records')
            
            # Populate the annotated_ids set (only count records that have been annotated)
            annotated_ids = set()
            for record in annotated_data:
                # Consider a record as annotated if it has a non-zero/non-empty adherence score
                if record.get('adherence_subject') or record.get('adherence_exercise_type') or \
                   record.get('adherence_difficulty') or record.get('adherence_grade_level') or \
                   record.get('adherence_modifier'):
                    annotated_ids.add(record.get('original_id'))
            
            print(f"✅ Loaded {len(annotated_data)} total records ({len(annotated_ids)} annotated)")
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
    # Ensure input is a string
    if text is None:
        return ""
    text = str(text)
    if text == "":
        return ""
    
    # Convert LaTeX equations to MathJax format. Wrap substitutions in try/except
    # to avoid failing on unexpected escape sequences in user-provided text.
    try:
        # Handle display equations \[ ... \]
        text = re.sub(r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}', r'$$\1$$', text, flags=re.DOTALL)
        text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r'$$\1$$', text, flags=re.DOTALL)

        # Handle align environments
        text = re.sub(r'\\begin\{align\*\}(.*?)\\end\{align\*\}', r'$$\begin{align}\1\end{align}$$', text, flags=re.DOTALL)
    except re.error:
        # Fall back to returning the raw text (escaped for markdown) if regex fails
        return _escape_backslashes_for_markdown(text)
    
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


def _escape_backslashes_for_markdown(s: object) -> str:
    """Return a Markdown-safe string by escaping backslashes.

    Gradio's Markdown component runs inspect.cleandoc on values which can
    raise errors for certain backslash sequences coming from LaTeX. Escape
    backslashes so the string is safe to pass to gr.Markdown and always
    return a string.
    """
    if s is None:
        return ""
    t = str(s)
    return t.replace("\\", "\\\\")


def _safe_int(val: object, default: Optional[int] = None) -> Optional[int]:
    """Safely convert a value to integer, handling None, empty strings, and invalid inputs.

    If the input is None or an empty string, this function returns ``None`` by
    default (or the provided ``default``), preserving empty/unknown adherence
    values instead of converting them to zero. This ensures empty adherence
    fields remain empty when displayed in the UI.

    Args:
        val: Value to convert (can be None, empty string, or any type)
        default: Default value to return if conversion fails (default: None)

    Returns:
        Converted integer, or ``default`` (often ``None``) if conversion fails
    """
    if val is None or val == "":
        return default
    try:
        return int(float(val))  # float() handles string representations of numbers
    except (ValueError, TypeError):
        return default


# Global variables
df = load_data()
current_sample = {}
# Initialize annotation tracking variables
load_existing_annotations()  # This will populate annotated_data and annotated_ids

def get_sample_list():
    """Get list of sample IDs for selection with annotation status"""
    sample_list = []
    
    if MODE == "ANNOTATION" and not df.empty:
        # Show samples from input data with annotation status
        for i, row in df.iterrows():
            # Mark annotated samples with ✅
            status_mark = "✅ " if i in annotated_ids else ""
            exercise_preview = row['exercise_type'][:50] if isinstance(row['exercise_type'], str) else str(row['exercise_type'])[:50]
            sample_list.append(f"{status_mark}ID {i}: {row['subject']} - {exercise_preview}...")
    else:
        # Show samples from annotated data
        if annotated_data:
            for record in annotated_data:
                original_id = record.get('original_id', '')
                new_subject = record.get('new_subject', record.get('original_subject', ''))
                new_exercise = record.get('new_exercise_type', record.get('original_exercise_type', ''))
                
                # Check if annotated
                is_annotated = bool(record.get('adherence_subject') or record.get('adherence_exercise_type') or 
                                   record.get('adherence_difficulty') or record.get('adherence_grade_level') or 
                                   record.get('adherence_modifier'))
                status_mark = "✅ " if is_annotated else ""
                
                exercise_preview = new_exercise[:50] if isinstance(new_exercise, str) else str(new_exercise)[:50]
                sample_list.append(f"{status_mark}ID {original_id}: {new_subject} - {exercise_preview}...")
    
    return sample_list

def load_sample(selected_id):
    """Load a sample based on selected ID"""
    if not selected_id:
        # Return None for adherence values so the UI displays them as empty
        return "", "", "", "", "", "", "", "", "", "", "", "**No sample selected**", None, None, None, None, None
    
    try:
        # Extract ID from the selection string
        clean_id = selected_id.replace("✅ ", "")
        id_num = int(clean_id.split(':')[0].replace('ID ', ''))
        
        global current_sample
        
        if MODE == "ANNOTATION" and not df.empty:
            # Load from input data
            row = df.iloc[id_num]
            current_sample = {
                'id': id_num,
                'instruction': row['instruction'],
                'subject': row['subject'],
                'exercise_type': row['exercise_type'],
                'difficulty': row['difficulty'],
                'grade_level': str(row['grade_level']),
                'modifier': row['modifier'],
                'generation': row['generation'],
            }
            
            # Check if this sample has previous annotations
            previous_annotation = None
            for annotation in annotated_data:
                if annotation['original_id'] == id_num:
                    previous_annotation = annotation
                    break
            
            # Prepare content for display with LaTeX rendering
            orig_instruction_md = _escape_backslashes_for_markdown(f"**Instruction:**\n\n{row['instruction']}")
            orig_generation_html = convert_latex_to_html(str(row['generation']))
            
            # Initialize adherence scores (use None to represent empty/unset)
            adherence_scores = [None, None, None, None, None]
            
            # If there's a previous annotation, use the edited values
            if previous_annotation:
                edit_subject = previous_annotation['new_subject']
                edit_exercise_type = previous_annotation['new_exercise_type']
                edit_difficulty = previous_annotation['new_difficulty']
                edit_grade_level = str(previous_annotation['new_grade_level'])
                edit_modifier = previous_annotation['new_modifier']
                edit_instruction_md = _escape_backslashes_for_markdown(f"**Updated Instruction:**\n\n{previous_annotation['new_instruction']}")
                
                # Load previous adherence scores (None means empty/unset)
                adherence_scores = [
                    _safe_int(previous_annotation.get('adherence_subject', None)),
                    _safe_int(previous_annotation.get('adherence_exercise_type', None)),
                    _safe_int(previous_annotation.get('adherence_difficulty', None)),
                    _safe_int(previous_annotation.get('adherence_grade_level', None)),
                    _safe_int(previous_annotation.get('adherence_modifier', None))
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
                edit_instruction_md = _escape_backslashes_for_markdown(f"**Instruction:** {row['instruction']}")
            
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
                # Return adherence scores (keep None for empty, don't convert to empty string)
                *adherence_scores
            )
        else:
            # Load from annotated data
            annotation = None
            for rec in annotated_data:
                if rec['original_id'] == id_num:
                    annotation = rec
                    break
            
            if not annotation:
                return "", "", "", "", "", "", "", "", "", "", "", "**No annotation found**", None, None, None, None, None
            
            current_sample = {
                'id': id_num,
                'instruction': annotation.get('original_instruction', ''),
                'subject': annotation.get('original_subject', ''),
                'exercise_type': annotation.get('original_exercise_type', ''),
                'difficulty': annotation.get('original_difficulty', ''),
                'grade_level': str(annotation.get('original_grade_level', '')),
                'modifier': annotation.get('original_modifier', ''),
                'generation': annotation.get('original_generation', ''),
            }
            
            # Show original values on left
            orig_instruction_md = _escape_backslashes_for_markdown(f"**Original Instruction:**\n\n{annotation.get('original_instruction', '')}")
            orig_generation_html = convert_latex_to_html(str(annotation.get('original_generation', '')))
            
            # Show current edited values on right
            edit_subject = annotation.get('new_subject', annotation.get('original_subject', ''))
            edit_exercise_type = annotation.get('new_exercise_type', annotation.get('original_exercise_type', ''))
            edit_difficulty = annotation.get('new_difficulty', annotation.get('original_difficulty', ''))
            edit_grade_level = str(annotation.get('new_grade_level', annotation.get('original_grade_level', '')))
            edit_modifier = annotation.get('new_modifier', annotation.get('original_modifier', ''))
            edit_instruction_md = _escape_backslashes_for_markdown(f"**Current Instruction:**\n\n{annotation.get('new_instruction', annotation.get('original_instruction', ''))}")

            # Ensure current_sample contains the edited/new values so saving without
            # further edits preserves them (prevents overwriting new_instruction with original)
            current_sample.update({
                'new_instruction': annotation.get('new_instruction', annotation.get('original_instruction', '')),
                'new_subject': annotation.get('new_subject', annotation.get('original_subject', '')),
                'new_exercise_type': annotation.get('new_exercise_type', annotation.get('original_exercise_type', '')),
                'new_difficulty': annotation.get('new_difficulty', annotation.get('original_difficulty', '')),
                'new_grade_level': annotation.get('new_grade_level', annotation.get('original_grade_level', '')),
                'new_modifier': annotation.get('new_modifier', annotation.get('original_modifier', '')),
            })
            
            # Load adherence scores (None means empty/unset)
            adherence_scores = [
                _safe_int(annotation.get('adherence_subject', None)),
                _safe_int(annotation.get('adherence_exercise_type', None)),
                _safe_int(annotation.get('adherence_difficulty', None)),
                _safe_int(annotation.get('adherence_grade_level', None)),
                _safe_int(annotation.get('adherence_modifier', None))
            ]

            return (
                annotation.get('original_subject', ''),
                annotation.get('original_exercise_type', ''),
                annotation.get('original_difficulty', ''),
                str(annotation.get('original_grade_level', '')),
                annotation.get('original_modifier', ''),
                orig_instruction_md,
                orig_generation_html,
                # Editable values
                edit_subject,
                edit_exercise_type,
                edit_difficulty,
                edit_grade_level,
                edit_modifier,
                edit_instruction_md,
                # Adherence scores (keep None for empty, don't convert to empty string)
                *adherence_scores
            )
    except Exception as e:
        print(f"Error loading sample: {e}")
        return "", "", "", "", "", "**No data available**", "", "", "", "", "", "**No data available**", None, None, None, None, None

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
    """Save the annotated data with adherence scores.
    
    When saving:
    1. Update/add the current sample annotation
    2. If using input-data mode, ensure ALL samples (annotated + unannotated) are in the output
    """
    global current_sample, annotated_data, annotated_ids
    
    if not current_sample:
        return "No sample loaded. Please select a sample first.", "", gr.Dropdown(choices=get_sample_list())
    
    # Validate adherence scores with new constraints
    try:
        # Allow empty/unset values (None) — treat as not provided
        # Subject adherence: if provided, must be 0 or 1
        subject_val: Optional[int]
        if adherence_subject is None or adherence_subject == "":
            subject_val = None
        else:
            subject_val = int(adherence_subject)
            if subject_val not in (0, 1):
                return "Subject adherence must be 0 or 1.", "", gr.Dropdown(choices=get_sample_list())

        # Other adherence scores: if provided, must be integers between 0-4
        # Cast values to int (int(float(x))) to accept numeric strings like "2" or "2.0"
        other_vals = []
        for raw in (adherence_exercise, adherence_difficulty, adherence_grade, adherence_modifier):
            if raw is None or raw == "":
                other_vals.append(None)
            else:
                # Convert to int via float to tolerate inputs like "2.0" and numeric widgets
                val_int = int(float(raw))
                if val_int < 0 or val_int > 4:
                    return (
                        "Exercise, Difficulty, Grade, and Modifier adherence must be integers between 0-4.",
                        "",
                        gr.Dropdown(choices=get_sample_list()),
                    )
                other_vals.append(val_int)

        adherence_scores = [subject_val, other_vals[0], other_vals[1], other_vals[2], other_vals[3]]
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
        'adherence_subject': adherence_scores[0] if adherence_scores[0] is not None else "",
        'adherence_exercise_type': adherence_scores[1] if adherence_scores[1] is not None else "",
        'adherence_difficulty': adherence_scores[2] if adherence_scores[2] is not None else "",
        'adherence_grade_level': adherence_scores[3] if adherence_scores[3] is not None else "",
        'adherence_modifier': adherence_scores[4] if adherence_scores[4] is not None else "",
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
    
    # If in ANNOTATION mode with input data, ensure ALL samples are in annotated_data
    # (including unannotated ones)
    if MODE == "ANNOTATION" and not df.empty:
        print("💾 Consolidating all samples into annotation file...")
        for i, row in df.iterrows():
            # Check if this sample already has an annotation
            has_annotation = any(rec['original_id'] == i for rec in annotated_data)
            
            if not has_annotation:
                # Create an unannotated record for this sample
                unannotated_record = {
                    'original_id': i,
                    'original_instruction': row['instruction'],
                    'original_subject': row['subject'],
                    'original_exercise_type': row['exercise_type'],
                    'original_difficulty': row['difficulty'],
                    'original_grade_level': row['grade_level'],
                    'original_modifier': row['modifier'],
                    'original_generation': row['generation'],
                    'new_instruction': '',
                    'new_subject': '',
                    'new_exercise_type': '',
                    'new_difficulty': '',
                    'new_grade_level': '',
                    'new_modifier': '',
                    'adherence_subject': '',
                    'adherence_exercise_type': '',
                    'adherence_difficulty': '',
                    'adherence_grade_level': '',
                    'adherence_modifier': '',
                    'annotation_timestamp': ''
                }
                annotated_data.append(unannotated_record)
        
        # Sort by original_id for consistency
        annotated_data.sort(key=lambda x: x['original_id'])
    
    # Save to file
    annotated_df = pd.DataFrame(annotated_data)
    output_file = args.annotated_data
    
    # Ensure adherence columns are saved as integers, not floats
    # (pandas may infer float type when reading CSV with empty cells)
    adherence_cols = ['adherence_subject', 'adherence_exercise_type', 'adherence_difficulty', 
                      'adherence_grade_level', 'adherence_modifier']
    for col in adherence_cols:
        if col in annotated_df.columns:
            # Convert any float values to int, preserve empty strings
            annotated_df[col] = annotated_df[col].apply(
                lambda x: int(x) if isinstance(x, (int, float)) and x != '' else x
            )
    
    # Preserve column order from original file if it exists
    if os.path.exists(output_file):
        try:
            # Read the existing file to get column order
            original_df = pd.read_csv(output_file)
            original_columns = original_df.columns.tolist()
            # Ensure all original columns are in the new dataframe
            for col in original_columns:
                if col not in annotated_df.columns:
                    annotated_df[col] = ''
            # Reorder to match original column order, then add any new columns
            cols_to_use = [col for col in original_columns if col in annotated_df.columns]
            new_cols = [col for col in annotated_df.columns if col not in cols_to_use]
            annotated_df = annotated_df[cols_to_use + new_cols]
        except Exception as e:
            print(f"⚠️  Could not preserve column order: {e}. Using default order.")
    
    annotated_df.to_csv(output_file, index=False)
    
    # Update the dropdown with new annotation status
    updated_choices = get_sample_list()

    # Find the current sample in the updated choices
    current_sample_with_mark = None
    for choice in updated_choices:
        if f"ID {current_sample['id']}:" in choice:
            current_sample_with_mark = choice
            break

    total_samples = len(df) if not df.empty else len(annotated_data)
    progress_pct = (len(annotated_ids) / total_samples * 100) if total_samples > 0 else 0

    return (
        f"✅ Annotation {action}! Progress: {len(annotated_ids)}/{total_samples} ({progress_pct:.1f}%)", 
        f"Saved to: {output_file}",
        gr.Dropdown(choices=updated_choices, value=current_sample_with_mark)
    )

def next_sample(current_dropdown_value):
    """Move to the next sample in the dataset"""
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
    
    # Mode indicator
    mode_indicator = "🔄 REVIEW MODE" if MODE == "REVIEW" else "✏️ ANNOTATION MODE"
    gr.Markdown(f"# 📊 Dataset Annotation Tool ({mode_indicator})")
    
    if MODE == "ANNOTATION":
        gr.Markdown("Select a sample from the list below to start annotation. ✅ indicates already annotated samples.")
        gr.Markdown("**Note:** When you save, ALL samples (annotated + unannotated) are saved to the annotation file. You can review them later without the original data.")
    else:
        gr.Markdown("**REVIEW MODE:** Edit existing annotations. ✅ indicates fully annotated samples.")
    
    # Get initial sample data
    sample_choices = get_sample_list()
    initial_sample = sample_choices[0] if sample_choices else None
    initial_data = load_sample(initial_sample) if initial_sample else ("", "", "", "", "", "", "", "", "", "", "", "**No sample selected**", None, None, None, None, None)
    
    with gr.Row():
        # Sample selection
        sample_dropdown = gr.Dropdown(
            choices=sample_choices,
            label="📋 Select Sample",
            info="Choose a sample to annotate/review",
            interactive=True,
            value=initial_sample
        )

    # Workspace area
    with gr.Row():
        # Left panel - Original data (read-only)
        with gr.Column(scale=1):
            left_panel_title = "📖 Original Data" if MODE == "ANNOTATION" else "📖 Original Data (Reference)"
            gr.Markdown(f"### {left_panel_title}")
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
            right_panel_title = "✏️ Editable Data" if MODE == "ANNOTATION" else "✏️ Edit Values"
            gr.Markdown(f"### {right_panel_title}")
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
            save_btn = gr.Button("💾 Save Changes" if MODE == "REVIEW" else "💾 Save Annotation", variant="primary", size="lg")
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

    # Initialize form on page load to ensure adherence scores are properly displayed as empty
    # This fixes the issue where adherence scores show 0 on initial page load
    demo.load(
        fn=load_sample,
        inputs=[sample_dropdown],
        outputs=[orig_subject, orig_exercise_type, orig_difficulty, orig_grade_level, 
                orig_modifier, orig_instruction, orig_generation, edit_subject, edit_exercise_type, 
                edit_difficulty, edit_grade_level, edit_modifier, edit_instruction,
                adherence_subject, adherence_exercise, adherence_difficulty, adherence_grade, adherence_modifier]
    )

if __name__ == "__main__":
    # Check and display startup info
    if MODE == "ANNOTATION":
        if not os.path.exists(args.input_data):
            print(f"⚠️  Data file not found. Please ensure '{args.input_data}' exists.")
        else:
            print(f"📊 ANNOTATION MODE: Loaded {len(df)} samples from dataset")
            print(f"📂 Input data: {args.input_data}")
            print(f"💾 Output file: {args.annotated_data}")
            if len(annotated_data) > 0:
                print(f"🎯 Resumed with {len(annotated_data)} total records ({len(annotated_ids)} annotated)")
                total = len(df)
                pct = (len(annotated_ids) / total * 100) if total > 0 else 0
                print(f"📝 Progress: {len(annotated_ids)}/{total} samples annotated ({pct:.1f}%)")
            else:
                print(f"🆕 Starting fresh annotation session")
    else:
        total = len(annotated_data)
        annotated = len(annotated_ids)
        pct = (annotated / total * 100) if total > 0 else 0
        print(f"🔄 REVIEW MODE: Loaded {total} records ({annotated} annotated, {total - annotated} unannotated)")
        print(f"💾 File: {args.annotated_data}")
        print(f"📝 Progress: {annotated}/{total} samples annotated ({pct:.1f}%)")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 