import re
import os


from semanticsilm.main import DATA_DIR, SOURCE_DIR

def split_chapters(input_file, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the entire file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into chapters
    chapters = re.split(r'\n\n+CHAPTER \d+\n\n+', content)

    # Remove any leading/trailing whitespace from each chapter
    chapters = [chapter.strip() for chapter in chapters if chapter.strip()]

    # Write each chapter to a separate file
    for i, chapter in enumerate(chapters, 1):
        # Extract the chapter title
        title_match = re.match(r'([A-Z\s]+)\n\n', chapter)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = f"Chapter {i}"

        # Create the filename
        filename = f"chapter{i}.txt"
        filepath = os.path.join(output_dir, filename)

        # Write the chapter content to the file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f"CHAPTER {i}\n\n")
            file.write(f"{title}\n\n")
            file.write(chapter)

        print(f"Wrote {filename}")

def main():
    input_file = f"{SOURCE_DIR}/raw.txt"
    output_dir = DATA_DIR
    split_chapters(input_file, output_dir)

if __name__ == "__main__":
    main()


