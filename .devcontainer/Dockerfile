FROM ultralytics/ultralytics:8.0.216-cpu
COPY requirements.txt /tmp/
RUN pip install --quiet --no-cache-dir -r /tmp/requirements.txt
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspaces/solarathon-team5/
ENV PYTHONPATH=/workspaces/solarathon-team5/
ENTRYPOINT ["solara", "run", "solarathon.pages", "--host=0.0.0.0", "--port=80"]