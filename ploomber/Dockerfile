#FROM python:3.11
FROM ultralytics/ultralytics:8.0.216-cpu

#RUN pip install --no-cache-dir --upgrade pip
#RUN pip install --pre torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu
RUN mkdir wheels
COPY wheels/*.whl wheels
RUN pip install wheels/*.whl

ENTRYPOINT ["solara", "run", "solarathon.pages", "--host=0.0.0.0", "--port=80"]
