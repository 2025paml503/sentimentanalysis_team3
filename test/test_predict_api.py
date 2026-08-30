"""
test/test_predict_api.py
========================

Integration test suite for the sentiment prediction API.
Tests the /predict endpoint with a variety of real Amazon review samples.

Run:
    python test/test_predict_api.py

Requirements:
    - API server running on http://localhost:8000
    - Use: uvicorn serving.api:app --port 8000
"""

import json
import sys
from pathlib import Path

import requests

# Test data: 20+ real Amazon reviews with expected sentiment
TEST_REVIEWS = [
    {
        "title": "Excellent product, highly recommended",
        "review_text": "This product exceeded all my expectations. Excellent quality, fast delivery, and great customer service. I would definitely buy again!",
        "expected_sentiment": "positive",
    },
    {
        "title": "Terrible quality, waste of money",
        "review_text": "Complete waste of money. The product broke after one week of normal use. Customer service was unhelpful and refused to provide a refund.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Great value for the price",
        "review_text": "Amazing product at an unbeatable price. Works perfectly as described. Shipping was fast and packaging was excellent.",
        "expected_sentiment": "positive",
    },
    {
        "title": "Disappointing experience",
        "review_text": "Ordered this item and waited weeks for delivery. When it arrived, it was damaged and completely unusable. Tried contacting support but got no response.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Exactly what I needed",
        "review_text": "This product is exactly as described. Good quality, arrived on time, and works perfectly. Definitely recommend to anyone looking for this type of item.",
        "expected_sentiment": "positive",
    },
    {
        "title": "Poor customer service",
        "review_text": "The product itself is okay, but the customer service was absolutely terrible. They were rude and unhelpful when I had an issue with my order.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Five stars all the way",
        "review_text": "Outstanding! This is the best product I have purchased in years. Quality is top-notch, delivery was super fast, and the price is unbeatable.",
        "expected_sentiment": "positive",
    },
    {
        "title": "Do not buy this",
        "review_text": "Do not waste your money on this product. It is cheap, poorly made, and stopped working after a few days. The company refuses to help.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Great purchase decision",
        "review_text": "Best purchase ever! The quality is exceptional, it arrived quickly, and I am completely satisfied with my purchase. Highly recommended!",
        "expected_sentiment": "positive",
    },
    {
        "title": "Not worth the money",
        "review_text": "This product is overpriced and underperforms. I expected much better quality given the high price. Very disappointed with this purchase.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Excellent customer service",
        "review_text": "Not only is the product fantastic, but the customer service team went above and beyond to help me. I'm very impressed with this company.",
        "expected_sentiment": "positive",
    },
    {
        "title": "Broken on arrival",
        "review_text": "The item arrived broken. Contacted support and they were unhelpful. This is my third order from this company and I'm done.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Fantastic purchase",
        "review_text": "This is an amazing product! Works perfectly, arrived ahead of schedule, and the packaging was excellent. Cannot ask for better quality.",
        "expected_sentiment": "positive",
    },
    {
        "title": "Defective product",
        "review_text": "Received a defective product. The company refused to replace it or give a refund. Very frustrated with this entire experience.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Absolutely love it",
        "review_text": "I love this product so much! It's even better than I expected. Great quality, excellent service, and arrived very quickly.",
        "expected_sentiment": "positive",
    },
    {
        "title": "Complete failure",
        "review_text": "This product failed completely. It doesn't work as advertised and the company won't help. Total waste of money.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Perfect for my needs",
        "review_text": "This product is perfect! Exactly what I was looking for. Great quality, fast delivery, and fantastic customer service. Highly satisfied!",
        "expected_sentiment": "positive",
    },
    {
        "title": "Extremely disappointed",
        "review_text": "Extremely disappointed with this purchase. Poor quality, arrived late, and customer service was non-existent. Will not buy again.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Best buy ever",
        "review_text": "This is the best purchase I've made in a long time. Excellent quality, amazing price, and super fast shipping. Highly recommend!",
        "expected_sentiment": "positive",
    },
    {
        "title": "Avoid this company",
        "review_text": "Avoid this company at all costs. Poor product quality, terrible customer service, and they refuse to help with returns or refunds.",
        "expected_sentiment": "negative",
    },
    {
        "title": "Exceeded expectations",
        "review_text": "This product exceeded my expectations in every way. Superior quality, beautiful packaging, and fantastic support team. A+++",
        "expected_sentiment": "positive",
    },
]


def test_predict_api(base_url="http://localhost:8000"):
    """
    Test the /predict endpoint with multiple review samples.

    Args:
        base_url (str): Base URL of the API server

    Returns:
        dict: Test results with pass/fail statistics
    """
    print("=" * 80)
    print("SENTIMENT PREDICTION API TEST SUITE")
    print("=" * 80)
    print(f"\nTarget API: {base_url}")
    print(f"Test Records: {len(TEST_REVIEWS)}\n")

    # Check health first
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        health_response.raise_for_status()
        health_data = health_response.json()
        print("✅ Health Check PASSED")
        print(f"   Status: {health_data.get('status')}")
        print(f"   Model Version: {health_data.get('model_version')}")
        print(f"   Vocabulary Size: {health_data.get('vocab_size')}\n")
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Cannot connect to API server")
        print(f"   Make sure the API is running: uvicorn serving.api:app --port 8000")
        return {"status": "failed", "error": "Connection refused"}
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return {"status": "failed", "error": str(e)}

    # Test predictions
    results = {
        "total": len(TEST_REVIEWS),
        "passed": 0,
        "failed": 0,
        "correct_sentiment": 0,
        "incorrect_sentiment": 0,
        "predictions": [],
    }

    print("-" * 80)
    print("PREDICTION TESTS")
    print("-" * 80)

    for idx, test_case in enumerate(TEST_REVIEWS, 1):
        try:
            payload = {
                "title": test_case["title"],
                "review_text": test_case["review_text"],
            }

            response = requests.post(
                f"{base_url}/predict",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            prediction = response.json()

            predicted_sentiment = prediction.get("sentiment")
            expected_sentiment = test_case["expected_sentiment"]
            confidence = prediction.get("confidence", 0)
            oov_rate = prediction.get("oov_rate", 0)

            # Check if prediction matches expected
            sentiment_match = predicted_sentiment == expected_sentiment

            # Log result
            status = "✅" if sentiment_match else "⚠️"
            print(f"\n{status} Test {idx}/{len(TEST_REVIEWS)}: {test_case['title'][:50]}...")
            print(f"   Expected: {expected_sentiment:8} | Predicted: {predicted_sentiment:8}")
            print(f"   Confidence: {confidence:.4f} | OOV Rate: {oov_rate:.4f}")
            print(f"   Latency: {prediction.get('latency_ms', 0):.2f}ms")

            if sentiment_match:
                results["passed"] += 1
                results["correct_sentiment"] += 1
            else:
                results["failed"] += 1
                results["incorrect_sentiment"] += 1

            results["predictions"].append({
                "title": test_case["title"],
                "expected": expected_sentiment,
                "predicted": predicted_sentiment,
                "confidence": confidence,
                "oov_rate": oov_rate,
                "match": sentiment_match,
            })

        except requests.exceptions.Timeout:
            print(f"\n❌ Test {idx}/{len(TEST_REVIEWS)}: TIMEOUT")
            print(f"   Title: {test_case['title'][:50]}...")
            results["failed"] += 1
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Test {idx}/{len(TEST_REVIEWS)}: REQUEST FAILED")
            print(f"   Title: {test_case['title'][:50]}...")
            print(f"   Error: {str(e)}")
            results["failed"] += 1
        except Exception as e:
            print(f"\n❌ Test {idx}/{len(TEST_REVIEWS)}: UNEXPECTED ERROR")
            print(f"   Title: {test_case['title'][:50]}...")
            print(f"   Error: {str(e)}")
            results["failed"] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']} ({100*results['passed']/results['total']:.1f}%)")
    print(f"Failed: {results['failed']} ({100*results['failed']/results['total']:.1f}%)")
    print(f"\nSentiment Accuracy: {results['correct_sentiment']}/{results['total']} correct")
    print(f"Misclassified: {results['incorrect_sentiment']}")

    # Calculate average confidence
    if results["predictions"]:
        avg_confidence = sum(p["confidence"] for p in results["predictions"]) / len(results["predictions"])
        avg_oov = sum(p["oov_rate"] for p in results["predictions"]) / len(results["predictions"])
        print(f"\nAverage Confidence: {avg_confidence:.4f}")
        print(f"Average OOV Rate: {avg_oov:.4f}")

    # Final status
    if results["failed"] == 0:
        print("\n✅ ALL TESTS PASSED!")
        results["status"] = "success"
    else:
        print(f"\n⚠️  {results['failed']} TEST(S) FAILED")
        results["status"] = "partial"

    print("=" * 80)

    return results


def test_batch_predict_api(base_url="http://localhost:8000"):
    """
    Test the /predict-batch endpoint with multiple reviews.

    Args:
        base_url (str): Base URL of the API server
    """
    print("\n" + "=" * 80)
    print("BATCH PREDICTION TEST")
    print("=" * 80)

    try:
        # Prepare batch payload (first 5 reviews)
        batch_reviews = [
            {
                "title": review["title"],
                "review_text": review["review_text"],
            }
            for review in TEST_REVIEWS[:5]
        ]

        payload = {"reviews": batch_reviews}

        response = requests.post(
            f"{base_url}/predict-batch",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()

        print(f"✅ Batch prediction successful!")
        print(f"   Processed: {result.get('count', 0)} reviews")
        print(f"   Total Latency: {result.get('total_latency_ms', 0):.2f}ms")
        print(f"   Results:")
        for idx, res in enumerate(result.get("results", []), 1):
            print(f"      {idx}. {res.get('sentiment')} (confidence: {res.get('confidence'):.4f})")

    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")


if __name__ == "__main__":
    try:
        api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

        # Run single prediction tests
        results = test_predict_api(base_url=api_url)

        # Run batch prediction test
        test_batch_predict_api(base_url=api_url)

        # Exit with appropriate code
        sys.exit(0 if results["status"] == "success" else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
