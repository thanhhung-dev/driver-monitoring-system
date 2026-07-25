from app.pipeline_factory import create_application
def main() -> None:
    """Build and run the Driver Monitoring System."""
    application = create_application()
    application.run()
    
if __name__ == "__main__":
    main()
