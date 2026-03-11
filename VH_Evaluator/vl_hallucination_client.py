#!/usr/bin/env python3
import base64
import os
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import argparse
import concurrent.futures
import time
from datetime import datetime
import glob
import statistics
import requests

class VLHallucinationClient:
    def __init__(self, api_url):
        """Initialize the evaluation client and connect to the deployed server."""
        self.api_url = api_url.rstrip('/')
        print(f"🔗 Connected to API: {self.api_url}")
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connectivity with the remote server."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Server connection verified")
            else:
                print(f"⚠️  Server returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not verify server connection: {e}")
            print(f"   Attempting to connect to {self.api_url}/v1/chat/completions...")
        
    def _encode_image(self, image_path):
        """Encode an image into base64."""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _load_reference_images(self, example_dir="./example"):
        """Load reference sample images to help the evaluator."""
        reference_images = {}
        example_files = sorted(glob.glob(os.path.join(example_dir, "example_*.jpg")) + 
                              glob.glob(os.path.join(example_dir, "example_*.png")))
        
        for i, example_file in enumerate(example_files[:3], 1):
            if os.path.exists(example_file):
                try:
                    reference_images[f"example_{i}"] = self._encode_image(example_file)
                except Exception as e:
                    print(f"Warning: Failed to load example {i}: {e}")
        
        return reference_images
    
    def _build_evaluation_prompt(self):
        """Construct the evaluation prompt."""
        prompt = """You are a professional image quality assessment expert specializing in evaluating over-optimization hallucination phenomena in images.

## Over-Optimization Hallucination Definition

Over-optimization hallucination includes the following types:

### 1. Grid Pattern Artifacts
- Regular grid textures, repetitive patterns, and other artifacts appear
- Usually caused by the internal structure of generative models

### 2. Texture Over-Enhancement
- Over-enhanced texture details that make the image look unnatural
- Surfaces appear overly rough or overly smooth

## Evaluation Task

Please evaluate the over-optimization hallucination phenomena in the following image:

**Image to Evaluate:** The image below

## Scoring Criteria

- **Over-optimization Level**: 0-5 points, 0 means no problem at all, 5 means extremely severe
  - detail_sharpness_score: Detail sharpening level (edge harshness, unnatural sharpening, high-frequency noise)
  - irrelevant_details_score: Irrelevant details level (extra details unrelated to prompt, over-decoration, information overload)
  - grid_pattern_score: Grid texture level
- **Overall Hallucination Level**: Average of the above three items
- **Confidence**: 0-1, indicating your certainty about the scoring

## Key Evaluation Points

1. **Detail Inspection**:
   - Are there unnecessary detail accumulations?
   - Are edges over-sharpened?
   - Are there elements unrelated to the original intent?

2. **Naturalness Assessment**: Does the image look natural overall, or does it appear "over-processed"?

3. **Artifact Detection**:
   - Look for grid patterns or repetitive textures
   - Check for over-enhanced or unnatural details
   - Identify any signs of over-optimization

## Reference Examples

The following are three reference examples showing different types of over-optimization hallucination:

### Example Comparison 1: Irrelevant Details Accumulation vs Moderate Details
- **Example 1 (Over-optimized)**: Added a large number of irrelevant detail elements, generated excessive decorative content, appearing chaotic and information-overloaded
- **Example 2 (Moderately optimized)**: Maintained moderate details without over-decoration, information is clear and well-organized
- **Key Difference**: Example 1's `irrelevant_details_score` should be significantly higher than Example 2's

### Example Comparison 2: Over-Sharpened Details vs Natural Sharpening
- **Example 1 (Over-optimized)**: Edges are over-sharpened, appearing harsh and unnatural, with obvious high-frequency noise and "over-refined" feeling
- **Example 3 (Moderately optimized)**: Edge sharpening is moderate, maintaining naturalness, details are clear but not harsh
- **Key Difference**: Example 1's `detail_sharpness_score` should be significantly higher than Example 3's

### Evaluation Guidance

When evaluating, please refer to these examples:
- When you see **large amounts of irrelevant details** similar to Example 1, increase `irrelevant_details_score`
- When you see **over-sharpened edges** similar to Example 1, increase `detail_sharpness_score`
- Example 2 and Example 3 represent more reasonable optimization levels and should receive lower hallucination scores

## Return Format (Must be valid JSON)

```json
{
    "image": {
        "has_over_optimization": true/false,
        "over_optimization_level": 0-5,
        "detail_sharpness_score": 0-5,
        "irrelevant_details_score": 0-5,
        "grid_pattern_score": 0-5,
        "overall_hallucination_score": 0-5,
        "confidence": 0-1,
        "detailed_analysis": "Detailed analysis description"
    }
}
```

Please carefully analyze the image and return the evaluation results in the above JSON format."""
        return prompt

    def process_image(self, image_path, output_file):
        """Process a single image by calling the remote API."""
        max_retries = 3
        attempt = 0
        result = None
        
        image_name = os.path.basename(image_path)
        
        while attempt < max_retries:
            try:
                attempt += 1
                
                # Encode image to base64
                image_b64 = self._encode_image(image_path)
                
                # Load reference images
                example_dir = "./example"
                if not os.path.exists(example_dir):
                    example_dir = os.path.join(os.path.dirname(__file__), "example")
                reference_images = self._load_reference_images(example_dir)
                
                # Build message content that includes reference images
                content = []
                
                # Append reference images with captions
                if reference_images:
                    content.append({"type": "text", "text": "## Reference Examples\n"})
                    for i in range(1, 4):
                        key = f"example_{i}"
                        if key in reference_images:
                            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{reference_images[key]}"}})
                            if i == 1:
                                content.append({"type": "text", "text": "Example 1 (Over-optimized)\n"})
                            elif i == 2:
                                content.append({"type": "text", "text": "Example 2 (Moderately optimized)\n"})
                            elif i == 3:
                                content.append({"type": "text", "text": "Example 3 (Moderately optimized)\n"})
                
                # Append evaluation prompt
                content.append({"type": "text", "text": self._build_evaluation_prompt()})
                
                # Append target image to evaluate
                content.append({"type": "text", "text": "\n## Image to Evaluate\n"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
                content.append({"type": "text", "text": f"Image: {image_name}\n"})
                
                # Build OpenAI-compatible message payload
                messages = [{
                    "role": "user",
                    "content": content
                }]
                
                # Call remote API
                response = requests.post(
                    f"{self.api_url}/v1/chat/completions",
                    json={
                        "messages": messages,
                        "max_tokens": 4096
                    },
                    timeout=600
                )
                
                if response.status_code != 200:
                    raise Exception(f"API returned status {response.status_code}: {response.text}")
                
                if not response.text:
                    raise Exception("Server returned empty response")
                
                response_json = response.json()
                output_text = response_json["choices"][0]["message"]["content"]
                
                # Try to parse JSON response
                try:
                    evaluation_result = json.loads(output_text)
                except json.JSONDecodeError:
                    # Attempt to extract JSON snippet from free-form text
                    import re
                    json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
                    if json_match:
                        evaluation_result = json.loads(json_match.group())
                    else:
                        raise ValueError("No valid JSON found in response")
                
                result = {
                    "image_name": image_name,
                    "image_path": image_path,
                    "evaluation": evaluation_result,
                    "attempt": attempt,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
                break
                
            except requests.exceptions.ConnectionError as e:
                if attempt == max_retries:
                    result = {
                        "image_name": image_name,
                        "image_path": image_path,
                        "error": f"Cannot connect to server at {self.api_url}. Make sure the VLM server is running.",
                        "attempt": attempt,
                        "success": False,
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"❌ Failed to process {image_name}: Connection error")
                else:
                    sleep_time = min(2 ** attempt, 10)
                    time.sleep(sleep_time)
            except requests.exceptions.Timeout:
                if attempt == max_retries:
                    result = {
                        "image_name": image_name,
                        "image_path": image_path,
                        "error": "Request timeout. Server is taking too long to respond.",
                        "attempt": attempt,
                        "success": False,
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"❌ Failed to process {image_name}: Timeout")
                else:
                    sleep_time = min(2 ** attempt, 10)
                    time.sleep(sleep_time)
            except Exception as e:
                if attempt == max_retries:
                    result = {
                        "image_name": image_name,
                        "image_path": image_path,
                        "error": str(e),
                        "attempt": attempt,
                        "success": False,
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"❌ Failed to process {image_name}: {str(e)}")
                else:
                    sleep_time = min(2 ** attempt, 10)
                    time.sleep(sleep_time)
        
        return result


def load_images(folder, limit=None):
    """Load all images under the given folder."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    images = []
    
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder, ext)))
        images.extend(glob.glob(os.path.join(folder, ext.upper())))
    
    images.sort()
    
    if limit:
        images = images[:limit]
    
    return images

def generate_summary_report(output_file, folder_name):
    """Generate a summary report from the JSONL results."""
    results = []
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("❌ No successful evaluations found")
        return
    
    # Aggregate stats
    overall_scores = []
    detail_sharpness_scores = []
    irrelevant_details_scores = []
    grid_pattern_scores = []
    
    for result in successful_results:
        eval_data = result.get('evaluation', {}).get('image', {})
        
        if eval_data:
            overall_scores.append(eval_data.get('overall_hallucination_score', 0))
            detail_sharpness_scores.append(eval_data.get('detail_sharpness_score', 0))
            irrelevant_details_scores.append(eval_data.get('irrelevant_details_score', 0))
            grid_pattern_scores.append(eval_data.get('grid_pattern_score', 0))
    
    # Statistical helpers
    def safe_avg(scores):
        return sum(scores) / len(scores) if scores else 0
    
    def safe_median(scores):
        return statistics.median(scores) if scores else 0
    
    def safe_stdev(scores):
        return statistics.stdev(scores) if len(scores) > 1 else 0
    
    # Compose summary
    report = {
        "evaluation_summary": {
            "total_images": len(results),
            "successful_images": len(successful_results),
            "failed_images": len(results) - len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "folder_name": folder_name
        },
        "hallucination_scores": {
            "avg_score": safe_avg(overall_scores),
            "median_score": safe_median(overall_scores),
            "min_score": min(overall_scores) if overall_scores else 0,
            "max_score": max(overall_scores) if overall_scores else 0,
            "std_dev": safe_stdev(overall_scores)
        },
        "detailed_metrics": {
            "detail_sharpness_avg": safe_avg(detail_sharpness_scores),
            "irrelevant_details_avg": safe_avg(irrelevant_details_scores),
            "grid_pattern_avg": safe_avg(grid_pattern_scores)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Persist summary
    report_file = output_file.replace('.jsonl', '_summary.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Print summary stats
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION SUMMARY REPORT")
    print(f"{'='*60}")
    print(f"\n📈 Overall Statistics:")
    print(f"  Folder: {folder_name}")
    print(f"  Total images: {report['evaluation_summary']['total_images']}")
    print(f"  Successful evaluations: {report['evaluation_summary']['successful_images']}")
    print(f"  Failed evaluations: {report['evaluation_summary']['failed_images']}")
    print(f"  Success rate: {report['evaluation_summary']['success_rate']:.2%}")
    
    print(f"\n🎯 Hallucination Scores (0-5 scale):")
    print(f"  Average: {report['hallucination_scores']['avg_score']:.2f}")
    print(f"  Median: {report['hallucination_scores']['median_score']:.2f}")
    print(f"  Range: [{report['hallucination_scores']['min_score']:.2f}, {report['hallucination_scores']['max_score']:.2f}]")
    print(f"  Std Dev: {report['hallucination_scores']['std_dev']:.2f}")
    
    print(f"\n📊 Detailed Metrics:")
    print(f"  Detail Sharpness: {report['detailed_metrics']['detail_sharpness_avg']:.2f}")
    print(f"  Irrelevant Details: {report['detailed_metrics']['irrelevant_details_avg']:.2f}")
    print(f"  Grid Pattern: {report['detailed_metrics']['grid_pattern_avg']:.2f}")
    
    print(f"\n{'='*60}")
    print(f"📄 Detailed report saved to: {report_file}")
    print(f"{'='*60}\n")

def process_image_wrapper(image_path, client):
    """Wrapper used by multiprocessing workers."""
    return client.process_image(image_path, None)

def main():
    parser = argparse.ArgumentParser(description="VL Hallucination Evaluation Client - Single Image Mode")
    parser.add_argument("--api_url", required=True,
                       help="API server URL (e.g., http://localhost:8000)")
    parser.add_argument("--folder", required=True,
                       help="Path to image folder")
    parser.add_argument("--folder_name", default=None,
                       help="Display name for folder")
    parser.add_argument("--output_path", default="./hallucination_results.jsonl",
                       help="Output file path")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of images to evaluate")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Number of worker processes")
    
    args = parser.parse_args()
    
    print(f"🔍 VL Hallucination Evaluation Client - Single Image Mode")
    print(f"API URL: {args.api_url}")
    print(f"Folder: {args.folder}")
    print(f"Output: {args.output_path}")
    print(f"Max workers: {args.max_workers}")
    if args.limit:
        print(f"Limit: {args.limit} images")
    
    # Initialize client
    print(f"\n🔗 Initializing client...")
    client = VLHallucinationClient(args.api_url)
    
    # Load images for evaluation
    print(f"\n📂 Loading images...")
    images = load_images(args.folder, args.limit)
    
    if not images:
        print("❌ No images found!")
        return
    
    print(f"✅ Found {len(images)} images")
    
    # Reset output file before writing results
    open(args.output_path, "w").close()
    
    # Run evaluations with multiprocessing
    print(f"\n🚀 Starting evaluation with {args.max_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        
        for image_path in images:
            futures.append(
                executor.submit(process_image_wrapper, image_path=image_path, client=client)
            )
        
        success_count = 0
        with tqdm(total=len(images), desc="Evaluating images...") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        success = result.get("success", False)
                        if success:
                            success_count += 1
                        else:
                            print(f"❌ {result['image_name']}: Evaluation failed - {result.get('error', 'Unknown error')}")
                        
                        # Append result to JSONL file
                        with open(args.output_path, "a", encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                finally:
                    pbar.update(1)
    
    # Summarize evaluation
    print(f"\n📊 Generating summary report...")
    generate_summary_report(
        args.output_path,
        args.folder_name or os.path.basename(args.folder)
    )
    
    print(f"📈 Final Statistics:")
    print(f"Total images: {len(images)}")
    print(f"Successful evaluations: {success_count}")
    print(f"Success rate: {success_count/len(images):.2%}")

if __name__ == "__main__":
    main()
