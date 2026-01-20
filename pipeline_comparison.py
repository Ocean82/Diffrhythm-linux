#!/usr/bin/env python3
"""
Compare different pipeline approaches for CPU-only systems
"""
import os


def show_comparison():
    """Show comparison of different pipeline approaches"""
    print("DiffRhythm Pipeline Approaches Comparison")
    print("=" * 50)
    print()

    approaches = {
        "1. Original Pipeline": {
            "file": "fix_infer.py",
            "memory": "~6-8GB (all models loaded)",
            "time": "15-20 minutes",
            "pros": ["Simple", "Standard approach"],
            "cons": [
                "High memory usage",
                "Can fail if RAM insufficient",
                "All-or-nothing",
            ],
        },
        "2. Optimized Pipeline": {
            "file": "optimized_cpu_inference.py",
            "memory": "~6-8GB (cached models)",
            "time": "8-15 minutes (fewer CFM steps)",
            "pros": ["Faster inference", "Better progress tracking", "Model caching"],
            "cons": ["Still high memory usage", "Can timeout on slow systems"],
        },
        "3. Segmented Pipeline": {
            "file": "segmented_pipeline.py",
            "memory": "~2-3GB (one model at a time)",
            "time": "10-18 minutes",
            "pros": ["Low memory usage", "Load/unload models", "CPU-friendly"],
            "cons": ["Model loading overhead", "More complex"],
        },
        "4. Checkpoint Pipeline": {
            "file": "checkpoint_pipeline.py",
            "memory": "~2-3GB (one model at a time)",
            "time": "10-18 minutes (can resume)",
            "pros": [
                "Resumable",
                "Never lose progress",
                "Lowest memory",
                "Most reliable",
            ],
            "cons": ["Disk I/O overhead", "More complex setup"],
        },
    }

    for name, details in approaches.items():
        print(f"{name}")
        print("-" * len(name))
        print(f"File: {details['file']}")
        print(f"Memory: {details['memory']}")
        print(f"Time: {details['time']}")
        print(f"Pros: {', '.join(details['pros'])}")
        print(f"Cons: {', '.join(details['cons'])}")
        print()

    print("RECOMMENDATIONS:")
    print("=" * 20)
    print()
    print("🥇 BEST FOR CPU-ONLY SYSTEMS:")
    print("   → checkpoint_pipeline.py")
    print("   • Lowest memory usage")
    print("   • Can resume if interrupted")
    print("   • Most reliable for slow systems")
    print()
    print("🥈 GOOD ALTERNATIVE:")
    print("   → segmented_pipeline.py")
    print("   • Lower memory than original")
    print("   • Simpler than checkpoint approach")
    print("   • Good balance of features")
    print()
    print("🥉 IF YOU HAVE ENOUGH RAM:")
    print("   → optimized_cpu_inference.py")
    print("   • Fastest inference")
    print("   • Good progress tracking")
    print("   • Traditional approach")
    print()

    print("QUICK START:")
    print("=" * 15)
    print("For your CPU-only system, try:")
    print("1. python checkpoint_pipeline.py  (most reliable)")
    print("2. python segmented_pipeline.py   (good alternative)")
    print("3. python optimized_cpu_inference.py  (if you have 8GB+ RAM)")


def check_system_resources():
    """Check system resources to recommend best approach"""
    print("\nSYSTEM RESOURCE CHECK:")
    print("=" * 25)

    try:
        import psutil

        # Check RAM
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)

        print(f"Total RAM: {total_gb:.1f} GB")
        print(f"Available RAM: {available_gb:.1f} GB")

        # Check CPU
        cpu_count = psutil.cpu_count()
        print(f"CPU cores: {cpu_count}")

        # Recommendations based on resources
        print("\nRECOMMENDATION BASED ON YOUR SYSTEM:")
        if available_gb >= 8:
            print("✓ You have enough RAM for the optimized pipeline")
            print("  → Try: python optimized_cpu_inference.py")
        elif available_gb >= 4:
            print("⚠ Limited RAM - use segmented approach")
            print("  → Try: python segmented_pipeline.py")
        else:
            print("⚠ Very limited RAM - use checkpoint approach")
            print("  → Try: python checkpoint_pipeline.py")

    except ImportError:
        print("psutil not available - install with: pip install psutil")
        print("Manual check: Look at your system's RAM usage")
        print("- 8GB+ available → optimized_cpu_inference.py")
        print("- 4-8GB available → segmented_pipeline.py")
        print("- <4GB available → checkpoint_pipeline.py")


def main():
    """Main comparison function"""
    show_comparison()

    try:
        check_system_resources()
    except Exception as e:
        print(f"Could not check system resources: {e}")

    print("\n" + "=" * 50)
    print("All approaches will work - choose based on your system and preferences!")


if __name__ == "__main__":
    main()
