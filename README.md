# Wound Band-Aid Detection

A lightweight OpenCV-based prototype that detects wound regions on arm images and digitally overlays a band-aid on top of them.

## Structure

```text
floccare-wound-bandaid-opencv/
│
├── assets/
│   └── bandaid.png
│
├── wound/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
│
├── output/
│
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Arulakash-J/wound-bandaid-opencv.git
   ```

   Move into the project directory:

   ```bash
   cd wound-bandaid-opencv
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Place the arm images you want to process inside the `wound/` folder.

Then run:

```bash
python main.py
```

The program will automatically process all supported images inside the `wound/` folder.

**Supported formats:**

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

## Technologies Used

- Python 3.12
- OpenCV
- NumPy

## Run the Project

After installation, simply run:

```bash
python main.py
```

The results will be available in the `output/` folder.
