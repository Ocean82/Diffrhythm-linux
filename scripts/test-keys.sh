#!/bin/bash
# Test which SSH key works for the EC2 instance

EC2_HOST="ec2-52-0-207-242.compute-1.amazonaws.com"
EC2_USER="ubuntu"
KEYS_DIRS=(
    "/mnt/c/Users/sammy/OneDrive/Desktop/AWS ITEMS"
    "/mnt/c/Users/sammy/OneDrive/Desktop/KEYS"
    "/mnt/d/BURNING-EMBERS"
    "/mnt/d/SERVER-SAVER"
    "/mnt/c/Users/sammy/.ssh"
)

echo "Testing SSH keys in multiple locations..."
echo ""

for KEYS_DIR in "${KEYS_DIRS[@]}"; do
    if [ -d "$KEYS_DIR" ]; then
        echo "Checking: $KEYS_DIR"
        
        # Test Burnt-Beats-KEY.pem first
        if [ -f "$KEYS_DIR/Burnt-Beats-KEY.pem" ]; then
            echo "  Testing: Burnt-Beats-KEY.pem"
            TEST_KEY="$HOME/test_key_$$"
            cp "$KEYS_DIR/Burnt-Beats-KEY.pem" "$TEST_KEY"
            chmod 600 "$TEST_KEY"
            
            if ssh -i "$TEST_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$EC2_USER@$EC2_HOST" "echo 'success'" 2>/dev/null | grep -q "success"; then
                echo "  ✓ SUCCESS! Burnt-Beats-KEY.pem works!"
                echo "  Full path: $KEYS_DIR/Burnt-Beats-KEY.pem"
                rm -f "$TEST_KEY"
                exit 0
            fi
            rm -f "$TEST_KEY"
        fi
        
        # Test all .pem files
        for key_file in "$KEYS_DIR"/*.pem; do
            if [ -f "$key_file" ]; then
                key_name=$(basename "$key_file")
                if [ "$key_name" != "Burnt-Beats-KEY.pem" ]; then
                    echo "  Testing: $key_name"
                    
                    TEST_KEY="$HOME/test_key_$$"
                    cp "$key_file" "$TEST_KEY"
                    chmod 600 "$TEST_KEY"
                    
                    if ssh -i "$TEST_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$EC2_USER@$EC2_HOST" "echo 'success'" 2>/dev/null | grep -q "success"; then
                        echo "  ✓ SUCCESS! This key works: $key_name"
                        echo "  Full path: $key_file"
                        rm -f "$TEST_KEY"
                        exit 0
                    else
                        echo "    ✗ Does not work"
                    fi
                    
                    rm -f "$TEST_KEY"
                fi
            fi
        done
        echo ""
    fi
done

echo "No working key found."
echo ""
echo "Please download Burnt-Beats-KEY.pem from AWS Console:"
echo "  AWS Console → EC2 → Key Pairs → Burnt-Beats-KEY → Download"
