output "ecr_repository_url" {
  value       = aws_ecr_repository.ml_api.repository_url
  description = "ECR repo URL for ml-api image"
}

output "ecs_cluster_name" {
  value       = aws_ecs_cluster.this.name
  description = "Name of the ECS cluster"
}

output "ml_api_security_group_id" {
  value       = aws_security_group.ml_api_sg.id
  description = "Security group id for ml-api service"
}
