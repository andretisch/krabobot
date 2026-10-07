import pypandoc
import os

def convert_md_to_docx(input_path, output_path):
    """
    Converts a Markdown file to a DOCX file using Pandoc.
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            return False, f"Input file not found: {input_path}"
        
        # Convert the file
        pypandoc.convert_file(input_path, 'docx', outputfile=output_path)
        
        # Verify if output file was actually created
        if os.path.exists(output_path):
            return True, f"Successfully converted {input_path} to {output_path}"
        else:
            return False, f"Pandoc reported success, but output file was not created at {output_path}"
            
    except Exception as e:
        return False, f"Conversion failed: {str(e)}"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    success, message = convert_md_to_docx(args.input, args.output)
    print(message)
    exit(0 if success else 1)
