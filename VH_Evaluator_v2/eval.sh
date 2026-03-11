python vl_hallucination_client.py \
  --api_url http://localhost:8000 \
  --folder  $1 \
  --output_path $2 \
  --limit 400 \
  --preprocessing laplacian clahe bilateral