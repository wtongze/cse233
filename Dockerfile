FROM python:3.10-bookworm
WORKDIR /app
COPY . .
RUN apt-get install git
RUN chmod +x install.sh
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN ./install.sh
CMD python3 red_evaluation.py
