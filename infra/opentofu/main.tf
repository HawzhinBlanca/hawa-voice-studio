# OpenTofu / Terraform Configuration for Hawa Sorani Voice Studio
terraform {
  required_version = ">= 1.6.0"
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Cloudflare R2 Bucket for Voice Assets & Checkpoints
resource "cloudflare_r2_bucket" "voice_assets" {
  account_id = var.cloudflare_account_id
  name       = "hawa-sorani-voice-assets"
  location   = "EEUR"
}

# Output Storage Endpoint
output "r2_bucket_name" {
  value = cloudflare_r2_bucket.voice_assets.name
}
