# AWS (DRAFT)

> **This is a draft.** The script and setup are untested end-to-end on a live instance.
> Blocked on AWS quota approval before we can validate the full flow.

## Status: Blocked on vCPU quota

The `run_aws.sh` script is fully working (builds Docker image, pushes to ECR, resolves AMI, launches EC2).
However, the AWS account (`113892881014`) has a **vCPU limit of 0** for G-type instances in `us-east-1`,
so the EC2 launch fails with `VcpuLimitExceeded`.

A quota increase request has been submitted:
- **Quota**: "Running On-Demand G and VT instances" (`L-DB2E81BA`)
- **Requested**: 4 vCPUs (enough for 1x `g6.xlarge` / L4)
- **Request ID**: `061d2aeaa3b54c2c97e66f6619160986zZWiuazq`
- **Region**: `us-east-1`

Check status:
```shell
aws service-quotas get-requested-service-quota-change \
  --region us-east-1 \
  --request-id 061d2aeaa3b54c2c97e66f6619160986zZWiuazq \
  --query "RequestedQuota.Status" --output text
```

This request needs to be **approved by Amazon** (not us — it's on their side).
It typically takes 30 minutes to a few hours, but can take up to 24 hours for new accounts.
If it stalls, you can check or re-request via the
[Service Quotas console](https://us-east-1.console.aws.amazon.com/servicequotas/home/services/ec2/quotas/L-DB2E81BA).

## Run

Once the quota is approved:

```shell
chmod +x run_aws.sh

S3_PATH=s3://sign-ml-datasets/videos/256x256 \
  S3_SCRATCH=s3://sign-ml-datasets/scratch \
  SECURITY_GROUP_ID=sg-00a6b5fdf760806b7 \
  KEY_NAME=cosmos-training \
  GPU=L4 GPUS=1 HOURS=1 \
  ./run_aws.sh
```

## SSH

```shell
ssh -i ~/.ssh/cosmos-training.pem ubuntu@<PUBLIC_IP>
tail -f /var/log/cosmos-training.log
```

## Notes

- **No IAM instance profile** is attached — the instance won't have AWS credentials for ECR/S3 access.
  This is fine for a POC launch test. For real training, create an instance profile with S3 + ECR permissions.
- SSH ingress (port 22) has been added to the default security group (`sg-00a6b5fdf760806b7`).
- Key pair `cosmos-training` was created; private key is at `~/.ssh/cosmos-training.pem`.
