import logging
from logging.config import fileConfig

from flask import current_app
from alembic import context

# This is the Alembic Config object, which provides
# access to values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)
logger = logging.getLogger('alembic.env')


def get_engine():
    """
    Get the SQLAlchemy engine instance from Flask-Migrate's db extension.
    Supports both Flask-SQLAlchemy < 3 and >= 3.
    """
    try:
        # For Flask-SQLAlchemy < 3 and Alchemical
        return current_app.extensions['migrate'].db.get_engine()
    except (TypeError, AttributeError):
        # For Flask-SQLAlchemy >= 3
        return current_app.extensions['migrate'].db.engine


def get_engine_url():
    """
    Return the database URL string, unmasking the password for Alembic config.
    """
    try:
        return get_engine().url.render_as_string(hide_password=False).replace('%', '%%')
    except AttributeError:
        return str(get_engine().url).replace('%', '%%')


# Add your model's MetaData object here for 'autogenerate' support.
# For example:
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
config.set_main_option('sqlalchemy.url', get_engine_url())
target_db = current_app.extensions['migrate'].db


def get_metadata():
    """
    Return the MetaData instance used by SQLAlchemy for migrations.
    """
    if hasattr(target_db, 'metadatas'):
        # For newer SQLAlchemy versions with multiple metadata support
        return target_db.metadatas[None]
    return target_db.metadata


def run_migrations_offline():
    """
    Run migrations in 'offline' mode.
    No DBAPI connection required.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=get_metadata(),
        literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """
    Run migrations in 'online' mode.
    Creates Engine and associates connection with context.
    """

    def process_revision_directives(context, revision, directives):
        """
        Prevent generation of empty migrations.
        """
        if getattr(config.cmd_opts, 'autogenerate', False):
            script = directives[0]
            if script.upgrade_ops.is_empty():
                directives[:] = []
                logger.info('No changes in schema detected.')

    conf_args = current_app.extensions['migrate'].configure_args
    if conf_args.get("process_revision_directives") is None:
        conf_args["process_revision_directives"] = process_revision_directives

    connectable = get_engine()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=get_metadata(),
            **conf_args
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

