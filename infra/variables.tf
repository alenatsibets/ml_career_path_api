variable "aws_region" {
  type        = string
  description = "AWS region to deploy to"
  default     = "eu-central-1"
}

variable "project_name" {
  type        = string
  description = "Prefix for resource names"
  default     = "ml-api"
}
