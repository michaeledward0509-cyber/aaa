#!/bin/bash

###########################################
# CPU Optimization Script for StreetVision
# Optimizes system for CPU-only mining with 16GB RAM
###########################################

echo "=========================================="
echo "CPU Optimization for StreetVision Miner"
echo "=========================================="
echo ""

# Check if running as root for system-level optimizations
if [ "$EUID" -eq 0 ]; then 
    echo "⚠️  Running as root. Some optimizations will be applied system-wide."
    SUDO=""
else
    echo "ℹ️  Running as user. Some optimizations require sudo."
    SUDO="sudo"
fi

echo ""
echo "1. Setting CPU thread limits (prevents memory spikes)..."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Make these persistent for current session
cat >> ~/.bashrc << 'EOL'

# CPU optimization for StreetVision
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
EOL

echo "✅ Thread limits set (4 threads per library)"

echo ""
echo "2. Checking swap space..."
SWAP_SIZE=$(free -g | grep Swap | awk '{print $2}')
if [ "$SWAP_SIZE" -lt 4 ]; then
    echo "⚠️  Swap space is less than 4GB. Creating 8GB swap file..."
    
    if [ -f /swapfile ]; then
        echo "   Swap file exists. Checking if it's active..."
        if ! swapon --show | grep -q /swapfile; then
            echo "   Activating existing swap file..."
            $SUDO swapon /swapfile
        else
            echo "   Swap file already active."
        fi
    else
        echo "   Creating new 8GB swap file (this may take a few minutes)..."
        $SUDO fallocate -l 8G /swapfile
        $SUDO chmod 600 /swapfile
        $SUDO mkswap /swapfile
        $SUDO swapon /swapfile
        
        # Make swap permanent
        if ! grep -q "/swapfile" /etc/fstab; then
            echo "/swapfile none swap sw 0 0" | $SUDO tee -a /etc/fstab
        fi
        echo "✅ Swap file created and activated"
    fi
else
    echo "✅ Swap space sufficient: ${SWAP_SIZE}GB"
fi

echo ""
echo "3. Checking PyTorch installation..."
if python3 -c "import torch; print('CPU:', torch.cuda.is_available() == False)" 2>/dev/null | grep -q "CPU: True"; then
    echo "✅ PyTorch CPU version detected"
else
    echo "⚠️  Consider installing CPU-optimized PyTorch:"
    echo "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
fi

echo ""
echo "4. Checking available RAM..."
TOTAL_RAM=$(free -g | grep Mem | awk '{print $2}')
AVAILABLE_RAM=$(free -g | grep Mem | awk '{print $7}')
echo "   Total RAM: ${TOTAL_RAM}GB"
echo "   Available RAM: ${AVAILABLE_RAM}GB"

if [ "$AVAILABLE_RAM" -lt 4 ]; then
    echo "⚠️  Low available RAM. Consider closing other applications."
else
    echo "✅ Sufficient RAM available"
fi

echo ""
echo "5. Checking miner.env configuration..."
if [ -f "miner.env" ]; then
    if grep -q "IMAGE_DETECTOR_DEVICE=cpu" miner.env; then
        echo "✅ miner.env configured for CPU"
    else
        echo "⚠️  miner.env not set to CPU. Updating..."
        sed -i 's/IMAGE_DETECTOR_DEVICE=.*/IMAGE_DETECTOR_DEVICE=cpu/' miner.env
        sed -i 's/VIDEO_DETECTOR_DEVICE=.*/VIDEO_DETECTOR_DEVICE=cpu/' miner.env
        echo "✅ Updated miner.env to use CPU"
    fi
else
    echo "⚠️  miner.env not found. Run ./setup_env.sh first"
fi

echo ""
echo "6. System recommendations..."
echo ""
echo "📋 Quick Checklist:"
echo "   [ ] Close unnecessary applications to free RAM"
echo "   [ ] Test on testnet (netuid 323) first"
echo "   [ ] Use lightweight model (ViT or YOLO-nano)"
echo "   [ ] Monitor with: htop or top"
echo "   [ ] Check logs for memory errors"
echo ""

echo "=========================================="
echo "Optimization Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Source your bashrc: source ~/.bashrc"
echo "2. Review miner.env configuration"
echo "3. Start miner: ./start_miner.sh"
echo ""
echo "For detailed CPU optimization guide, see:"
echo "   docs/CPU_Optimization_Guide.md"
echo ""
