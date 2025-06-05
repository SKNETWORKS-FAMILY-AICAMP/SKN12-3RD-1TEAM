import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import vector_manger as vm

logger = logging.getLogger(__name__)

class VectorDBUpdater:
    """
    Handles dynamic updates to VectorDB with new external data
    """
    
    def __init__(self):
        self.embedding_model = vm.get_embedding()
        self.category_to_db = vm.category_to_db
        self.update_log_file = self._get_update_log_path()
        
    def _get_update_log_path(self) -> Path:
        """Get path for update log file"""
        project_root = vm.get_project_root()
        log_dir = project_root / "data" / "db" / "updates"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "vectordb_updates.json"
    
    def add_documents_to_db(self, documents: List[Document], category: str) -> bool:
        """
        Add new documents to the appropriate VectorDB
        
        Args:
            documents: List of Document objects to add
            category: Category for the documents (관광지, 숙박, 대중교통)
            
        Returns:
            bool: Success status
        """
        try:
            if category not in self.category_to_db:
                logger.warning(f"Unknown category: {category}")
                return False
                
            db_name = self.category_to_db[category]
            
            # Load existing database
            existing_db = vm.load_db(db_name)
            
            # Create texts and metadatas for new documents
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add documents to existing database
            existing_db.add_texts(texts=texts, metadatas=metadatas)
            
            # Save updated database
            self._save_updated_db(existing_db, db_name)
            
            # Log the update
            self._log_update(category, len(documents), db_name)
            
            logger.info(f"Successfully added {len(documents)} documents to {db_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to DB: {str(e)}")
            return False
    
    def _save_updated_db(self, db: FAISS, db_name: str):
        """Save updated database to disk"""
        try:
            project_root = vm.get_project_root()
            db_path = project_root / "data" / "db" / "faiss" / db_name
            
            # Create backup of existing DB
            backup_path = project_root / "data" / "db" / "backups" / f"{db_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Save updated database
            db.save_local(str(db_path))
            
            # Update cache
            vm._db_cache[db_name] = db
            
            logger.info(f"Updated database saved: {db_name}")
            
        except Exception as e:
            logger.error(f"Error saving updated database: {str(e)}")
            raise
    
    def _log_update(self, category: str, doc_count: int, db_name: str):
        """Log database update for tracking"""
        try:
            # Load existing log
            if self.update_log_file.exists():
                with open(self.update_log_file, 'r', encoding='utf-8') as f:
                    update_log = json.load(f)
            else:
                update_log = {"updates": []}
            
            # Add new update entry
            update_entry = {
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "db_name": db_name,
                "documents_added": doc_count,
                "source": "external_api"
            }
            
            update_log["updates"].append(update_entry)
            
            # Keep only last 100 updates
            if len(update_log["updates"]) > 100:
                update_log["updates"] = update_log["updates"][-100:]
            
            # Save updated log
            with open(self.update_log_file, 'w', encoding='utf-8') as f:
                json.dump(update_log, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging update: {str(e)}")
    
    def get_update_history(self, category: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get update history for monitoring
        
        Args:
            category: Filter by category (optional)
            limit: Maximum number of entries to return
            
        Returns:
            List of update entries
        """
        try:
            if not self.update_log_file.exists():
                return []
                
            with open(self.update_log_file, 'r', encoding='utf-8') as f:
                update_log = json.load(f)
            
            updates = update_log.get("updates", [])
            
            # Filter by category if specified
            if category:
                updates = [u for u in updates if u.get("category") == category]
            
            # Sort by timestamp (newest first) and limit
            updates.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return updates[:limit]
            
        except Exception as e:
            logger.error(f"Error getting update history: {str(e)}")
            return []
    
    def create_documents_from_api_data(self, api_data: List[Dict[str, Any]], 
                                     category: str, user_query: str = "") -> List[Document]:
        """
        Convert API data to Document objects with enhanced metadata
        
        Args:
            api_data: Raw data from external API
            category: Category classification
            user_query: Original user query for context
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for item in api_data:
            # Create comprehensive content
            content_parts = []
            
            # Title/Name
            if item.get('title'):
                content_parts.append(f"장소명: {item['title']}")
            
            # Address
            if item.get('addr1'):
                content_parts.append(f"주소: {item['addr1']}")
            
            # Contact
            if item.get('tel'):
                content_parts.append(f"연락처: {item['tel']}")
            
            # Pet information
            if item.get('pet_info'):
                content_parts.append(f"반려동물 동반 정보: {item['pet_info']}")
            
            # Additional information
            for key, value in item.items():
                if key not in ['title', 'addr1', 'tel', 'pet_info', 'contentid'] and value:
                    content_parts.append(f"{key}: {value}")
            
            content = "\n".join(content_parts)
            
            # Enhanced metadata
            metadata = item.copy()
            metadata.update({
                'category': category,
                'data_source': 'external_api',
                'added_timestamp': datetime.now().isoformat(),
                'original_query': user_query,
                'content_length': len(content),
                'has_pet_info': bool(item.get('pet_info'))
            })
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def cleanup_old_external_data(self, days_old: int = 30) -> int:
        """
        Remove external data older than specified days from VectorDB
        This is a placeholder for future implementation
        
        Args:
            days_old: Remove data older than this many days
            
        Returns:
            Number of documents removed
        """
        # This would require rebuilding the entire FAISS index
        # For now, just log the request
        logger.info(f"Cleanup requested for data older than {days_old} days")
        logger.info("Note: Full cleanup implementation requires FAISS index rebuild")
        return 0
    
    def get_db_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current VectorDB state
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        
        try:
            for category, db_name in self.category_to_db.items():
                try:
                    db = vm.load_db(db_name)
                    
                    # Basic stats
                    stats[category] = {
                        'db_name': db_name,
                        'total_documents': db.index.ntotal if hasattr(db, 'index') else 'unknown',
                        'is_loaded': db_name in vm._db_cache
                    }
                    
                except Exception as e:
                    stats[category] = {
                        'db_name': db_name,
                        'error': str(e),
                        'is_loaded': False
                    }
                    
        except Exception as e:
            logger.error(f"Error getting DB stats: {str(e)}")
            
        return stats


# Convenience function
def update_vectordb_with_external_data(api_data: List[Dict[str, Any]], 
                                     category: str, 
                                     user_query: str = "") -> bool:
    """
    Convenience function to update VectorDB with external API data
    
    Args:
        api_data: Data from external API
        category: Category for the data
        user_query: Original user query for context
        
    Returns:
        Success status
    """
    updater = VectorDBUpdater()
    documents = updater.create_documents_from_api_data(api_data, category, user_query)
    return updater.add_documents_to_db(documents, category)


# Example usage
if __name__ == "__main__":
    # Example of how to use the updater
    updater = VectorDBUpdater()
    
    # Example API data (simulated)
    sample_api_data = [
        {
            'title': '강릉 애견동반 펜션',
            'addr1': '강원도 강릉시 어딘가',
            'tel': '033-123-4567',
            'pet_info': '소형견만 가능, 추가요금 있음',
            'contentid': '12345'
        }
    ]
    
    # Convert to documents and add to DB
    documents = updater.create_documents_from_api_data(
        sample_api_data, 
        "숙박", 
        "강릉 반려동물 숙박"
    )
    
    # Print document for verification
    print("Created document:")
    print(f"Content: {documents[0].page_content}")
    print(f"Metadata: {documents[0].metadata}")
    
    # Get DB stats
    print("\nDatabase Statistics:")
    stats = updater.get_db_stats()
    for category, stat in stats.items():
        print(f"{category}: {stat}") 