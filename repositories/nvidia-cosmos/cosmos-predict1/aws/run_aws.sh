#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Cosmos-Predict1 AWS Training Launcher
#
# Builds & pushes Docker image to ECR, launches a GPU EC2 instance,
# mounts S3 data, runs torchrun training, syncs results, auto-terminates.
# ──────────────────────────────────────────────────────────────────────────────

# ── Configuration (override via environment) ──────────────────────────────────
S3_PATH="${S3_PATH:?'S3_PATH is required (e.g. s3://bucket/videos)'}"
S3_SCRATCH="${S3_SCRATCH:?'S3_SCRATCH is required (e.g. s3://bucket/scratch)'}"
IAM_INSTANCE_PROFILE="${IAM_INSTANCE_PROFILE:-}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:?'SECURITY_GROUP_ID is required'}"
KEY_NAME="${KEY_NAME:?'KEY_NAME is required'}"

REGION="${REGION:-us-east-1}"
GPUS="${GPUS:-1}"
GPU="${GPU:-A100}"
HOURS="${HOURS:-1}"

ECR_REPO="cosmos-predict1"
IMAGE_TAG="latest"

# ── Paths (relative to reproduction root) ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"  # reproduction root

# ── 1. Validate inputs ───────────────────────────────────────────────────────
echo "==> Validating inputs..."

if [[ ! "$GPU" =~ ^(A100|H100|L4)$ ]]; then
  echo "ERROR: GPU must be A100, H100, or L4 (got: $GPU)" >&2
  exit 1
fi

if ! [[ "$GPUS" =~ ^[1-8]$ ]]; then
  echo "ERROR: GPUS must be 1-8 (got: $GPUS)" >&2
  exit 1
fi

if ! [[ "$HOURS" =~ ^[0-9]+$ ]] || (( HOURS < 1 )); then
  echo "ERROR: HOURS must be a positive integer (got: $HOURS)" >&2
  exit 1
fi

if [[ ! "$S3_PATH" =~ ^s3:// ]]; then
  echo "ERROR: S3_PATH must start with s3:// (got: $S3_PATH)" >&2
  exit 1
fi

if [[ ! "$S3_SCRATCH" =~ ^s3:// ]]; then
  echo "ERROR: S3_SCRATCH must start with s3:// (got: $S3_SCRATCH)" >&2
  exit 1
fi

# ── 2. Resolve instance type ───────────────────────────────────────────────────
echo "==> Resolving instance type for ${GPUS}x ${GPU}..."

resolve_instance_type() {
  local gpu="$1" gpus="$2"
  case "${gpu}_${gpus}" in
    L4_1)   echo "g6.xlarge" ;;
    L4_2)   echo "g6.12xlarge" ;;
    L4_4)   echo "g6.48xlarge" ;;
    L4_8)   echo "g6e.48xlarge" ;;
    A100_8) echo "p4d.24xlarge" ;;
    H100_1) echo "p5.4xlarge" ;;
    H100_8) echo "p5.48xlarge" ;;
    *)      echo "" ;;
  esac
}

INSTANCE_TYPE=$(resolve_instance_type "$GPU" "$GPUS")

if [[ -z "$INSTANCE_TYPE" ]]; then
  echo "ERROR: No known instance type mapping for ${GPUS}x ${GPU}" >&2
  echo "Supported: L4 (1,2,4,8), A100 (8), H100 (1,8)" >&2
  exit 1
fi


echo "    Instance type: $INSTANCE_TYPE"

# ── 3. Query price ───────────────────────────────────────────────────────────
echo "==> Querying on-demand price for ${INSTANCE_TYPE}..."

PRICE_USD=$(aws pricing get-products \
  --region us-east-1 \
  --service-code AmazonEC2 \
  --filters \
    "Type=TERM_MATCH,Field=instanceType,Value=${INSTANCE_TYPE}" \
    "Type=TERM_MATCH,Field=regionCode,Value=${REGION}" \
    "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
    "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
    "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
    "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
  --query "PriceList" \
  --output json \
| jq -r '
    if length==0 then "N/A"
    else
      (.[0] | fromjson) as $p
      | [ $p | .. | objects
          | select(has("pricePerUnit") and .pricePerUnit.USD? != null)
          | .pricePerUnit.USD
        ] | .[0] // "N/A"
    end
  ')

if [[ "$PRICE_USD" == "N/A" ]]; then
  echo "WARNING: Could not retrieve pricing for ${INSTANCE_TYPE} in ${REGION}."
  echo "         Proceeding without cost estimate."
  TOTAL_EST="unknown"
else
  TOTAL_EST=$(jq -nr --arg p "$PRICE_USD" --argjson h "$HOURS" '$p|tonumber * $h | . * 100 | round / 100')
  PRICE_DISPLAY=$(printf "%.2f" "$PRICE_USD")
  printf "    Cost: \$%s/hr × %d hrs = \$%s estimated total\n" "$PRICE_DISPLAY" "$HOURS" "$TOTAL_EST"
fi

read -rp "Proceed? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

# ── 4. Build & push Docker image to ECR ──────────────────────────────────────
echo "==> Building Docker image..."
docker build -t "${ECR_REPO}:${IMAGE_TAG}" \
  -f "$SCRIPT_DIR/Dockerfile" \
  "$REPO_ROOT"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
FULL_IMAGE="${ECR_URI}/${ECR_REPO}:${IMAGE_TAG}"

echo "==> Ensuring ECR repository exists..."
aws ecr describe-repositories \
  --region "$REGION" \
  --repository-names "$ECR_REPO" &>/dev/null \
|| aws ecr create-repository \
  --region "$REGION" \
  --repository-name "$ECR_REPO" \
  --query "repository.repositoryUri" \
  --output text

echo "==> Pushing image to ECR..."
aws ecr get-login-password --region "$REGION" \
| docker login --username AWS --password-stdin "$ECR_URI"

docker tag "${ECR_REPO}:${IMAGE_TAG}" "$FULL_IMAGE"
docker push "$FULL_IMAGE"

# ── 5. Resolve AMI ───────────────────────────────────────────────────────────
echo "==> Resolving latest Deep Learning AMI..."

AMI_ID=$(aws ec2 describe-images \
  --region "$REGION" \
  --owners amazon \
  --filters \
    "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu*" \
    "Name=state,Values=available" \
    "Name=architecture,Values=x86_64" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --output text)

echo "    AMI: $AMI_ID"

# ── 6. Generate user-data ────────────────────────────────────────────────────
echo "==> Generating user-data script..."

MINUTES=$(( HOURS * 60 ))

# S3 data bucket and prefix for mounting (read-only)
S3_DATA_BUCKET=$(echo "$S3_PATH" | sed 's|s3://||' | cut -d/ -f1)
S3_DATA_PREFIX=$(echo "$S3_PATH" | sed "s|s3://${S3_DATA_BUCKET}/\?||")

# S3 scratch bucket and prefix for mounting (read-write: checkpoints + outputs)
S3_SCRATCH_BUCKET=$(echo "$S3_SCRATCH" | sed 's|s3://||' | cut -d/ -f1)
S3_SCRATCH_PREFIX=$(echo "$S3_SCRATCH" | sed "s|s3://${S3_SCRATCH_BUCKET}/\?||")

USERDATA=$(cat <<USERDATA_EOF
#!/bin/bash
set -ex

# Safety net: auto-shutdown after ${MINUTES} minutes
shutdown -h +${MINUTES} "Auto-termination after ${HOURS}h"

exec > >(tee /var/log/cosmos-training.log) 2>&1
echo "=== Cosmos-Predict1 Training — \$(date -u) ==="

# ── Install mountpoint-s3 ────────────────────────────────────────────────
apt-get update -y
apt-get install -y wget
wget -q https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
dpkg -i mount-s3.deb || apt-get install -yf
rm mount-s3.deb

# ── Mount S3 dataset (read-only) ─────────────────────────────────────────
mkdir -p /mnt/s3-data
if [[ -n "${S3_DATA_PREFIX}" ]]; then
  mount-s3 --read-only --prefix "${S3_DATA_PREFIX}/" "${S3_DATA_BUCKET}" /mnt/s3-data
else
  mount-s3 --read-only "${S3_DATA_BUCKET}" /mnt/s3-data
fi
echo "S3 dataset mounted at /mnt/s3-data"

# ── Mount S3 scratch (read-write: checkpoints + outputs) ─────────────────
mkdir -p /mnt/scratch
if [[ -n "${S3_SCRATCH_PREFIX}" ]]; then
  mount-s3 --allow-write --prefix "${S3_SCRATCH_PREFIX}/" "${S3_SCRATCH_BUCKET}" /mnt/scratch
else
  mount-s3 --allow-write "${S3_SCRATCH_BUCKET}" /mnt/scratch
fi
mkdir -p /mnt/scratch/checkpoints
echo "S3 scratch mounted at /mnt/scratch"

# ── ECR login & pull image ───────────────────────────────────────────────
aws ecr get-login-password --region ${REGION} \
| docker login --username AWS --password-stdin ${ECR_URI}
docker pull ${FULL_IMAGE}

# ── Download HuggingFace pretrained checkpoints (cached in scratch) ──────
pip install -q huggingface_hub
for model in CV8x8x8-720p DV8x16x16-720p CV4x8x8-360p DV4x8x8-360p; do
  DEST="/mnt/scratch/checkpoints/Cosmos-Tokenize1-\${model}"
  if [ -d "\$DEST" ] && [ "\$(ls -A "\$DEST" 2>/dev/null)" ]; then
    echo "Checkpoint Cosmos-Tokenize1-\${model} already in scratch, skipping download"
  else
    echo "Downloading nvidia/Cosmos-Tokenize1-\${model} to scratch..."
    SNAP=\$(huggingface-cli download "nvidia/Cosmos-Tokenize1-\${model}" --cache-dir /tmp/hf-cache | tail -n 1)
    aws s3 sync "\$SNAP" "${S3_SCRATCH%/}/checkpoints/Cosmos-Tokenize1-\${model}/" --region ${REGION}
  fi
done

# ── Run training ─────────────────────────────────────────────────────────
echo "=== Starting training at \$(date -u) ==="
docker run --rm --gpus all \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/scratch/checkpoints:/workspace/checkpoints \
  -v /mnt/s3-data:/workspace/datasets/hdvila/videos/ \
  -e OUTPUT_ROOT=checkpoints \
  ${FULL_IMAGE} \
  torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS} \
    -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=Cosmos_Tokenize1_DV8x16x16_256p_HDVILA

echo "=== Training finished at \$(date -u) ==="

# ── Shutdown ─────────────────────────────────────────────────────────────
echo "All done. Outputs saved to ${S3_SCRATCH}"
shutdown -h now
USERDATA_EOF
)

# ── 7. Launch EC2 instance ────────────────────────────────────────────────────
echo "==> Launching ${INSTANCE_TYPE} in ${REGION}..."

IAM_PROFILE_FLAG=()
if [[ -n "$IAM_INSTANCE_PROFILE" ]]; then
  IAM_PROFILE_FLAG=(--iam-instance-profile "Name=${IAM_INSTANCE_PROFILE}")
fi

INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  ${IAM_PROFILE_FLAG[@]+"${IAM_PROFILE_FLAG[@]}"} \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cosmos-predict1-training},{Key=Project,Value=cosmos-predict1},{Key=AutoTerminate,Value=${HOURS}h}]" \
  --user-data "$USERDATA" \
  --query "Instances[0].InstanceId" \
  --output text)

echo "    Instance: $INSTANCE_ID"

# Wait for public IP
echo "==> Waiting for public IP..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text)

# ── 8. Print summary ─────────────────────────────────────────────────────────
TERMINATE_TIME=$(date -u -d "+${HOURS} hours" "+%Y-%m-%d %H:%M UTC" 2>/dev/null \
  || date -u -v+${HOURS}H "+%Y-%m-%d %H:%M UTC" 2>/dev/null \
  || echo "~${HOURS}h from now")

cat <<SUMMARY

════════════════════════════════════════════════════════════════════════════════
  Cosmos-Predict1 Training Launched
════════════════════════════════════════════════════════════════════════════════
  Instance ID:    $INSTANCE_ID
  Instance Type:  $INSTANCE_TYPE (${GPUS}x ${GPU})
  Public IP:      $PUBLIC_IP
  Region:         $REGION

  SSH:            ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}
  Logs:           ssh ... tail -f /var/log/cosmos-training.log

  Cost:           \$${PRICE_USD:-?}/hr × ${HOURS}h = \$${TOTAL_EST} estimated
  Auto-terminate: ${TERMINATE_TIME}

  S3 Data:        $S3_PATH
  S3 Scratch:     $S3_SCRATCH

  Terminate now:  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID
════════════════════════════════════════════════════════════════════════════════
SUMMARY
