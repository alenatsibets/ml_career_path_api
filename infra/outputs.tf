output "ml_ecr_repo_url" {
  value = aws_ecr_repository.ml.repository_url
}

output "ml_service_name" {
  value = aws_ecs_service.ml.name
}

output "ml_cloudmap_name" {
  value = "ml.ml.local"
}
