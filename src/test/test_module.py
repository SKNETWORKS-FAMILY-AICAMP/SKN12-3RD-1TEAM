import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from module import get_naver_map_link
from urllib.parse import quote

class TestModule(unittest.TestCase):
    def test_get_naver_map_link_basic(self):
        """기본 기능 테스트"""
        # 기본 장소명 테스트
        expected_url = f"https://map.naver.com/p/search/{quote('속초해수욕장')}"
        self.assertEqual(get_naver_map_link("속초해수욕장"), expected_url)
        
        # 공백이 포함된 장소명 테스트
        expected_url = f"https://map.naver.com/p/search/{quote('속초 해수욕장')}"
        self.assertEqual(get_naver_map_link("속초 해수욕장"), expected_url)
        
        # 특수문자가 포함된 장소명 테스트
        expected_url = f"https://map.naver.com/p/search/{quote('속초(해수욕장)')}"
        self.assertEqual(get_naver_map_link("속초(해수욕장)"), expected_url)
        
        # 긴 장소명 테스트
        test_place = "속초 멍스타그램 카페 애견동반 가능"
        expected_url = f"https://map.naver.com/p/search/{quote(test_place)}"
        self.assertEqual(get_naver_map_link(test_place), expected_url)

    def test_get_naver_map_link_with_city(self):
        """도시명 추가 기능 테스트"""
        # 도시명이 이미 포함된 경우
        self.assertEqual(
            get_naver_map_link("속초해수욕장", city="속초"),
            f"https://map.naver.com/p/search/{quote('속초해수욕장')}"
        )
        
        # 도시명이 포함되지 않은 경우
        self.assertEqual(
            get_naver_map_link("해수욕장", city="속초"),
            f"https://map.naver.com/p/search/{quote('속초 해수욕장')}"
        )

    def test_get_naver_map_link_error_handling(self):
        """에러 처리 테스트"""
        # None 입력 테스트
        with self.assertRaises(ValueError):
            get_naver_map_link(None)
        
        # 빈 문자열 테스트
        with self.assertRaises(ValueError):
            get_naver_map_link("")
        
        # 공백만 있는 문자열 테스트
        with self.assertRaises(ValueError):
            get_naver_map_link("   ")

    def test_get_naver_map_link_whitespace_handling(self):
        """공백 처리 테스트"""
        # 앞뒤 공백이 있는 경우
        self.assertEqual(
            get_naver_map_link("  속초해수욕장  "),
            f"https://map.naver.com/p/search/{quote('속초해수욕장')}"
        )
        
        # 도시명과 함께 사용할 때 공백 처리
        self.assertEqual(
            get_naver_map_link("  해수욕장  ", city="속초"),
            f"https://map.naver.com/p/search/{quote('속초 해수욕장')}"
        )

    def test_get_naver_map_link_real_places(self):
        """실제 장소명 테스트"""
        places = [
            "속초해수욕장",
            "속초 개밥주는 집",
            "펫프렌즈 속초점",
            "속초 장사항",
            "속초 멍스타그램 카페",
            "속초 엑스포타워",
            "속초 펫토이 마켓"
        ]
        
        for place in places:
            expected_url = f"https://map.naver.com/p/search/{quote(place)}"
            result_url = get_naver_map_link(place)
            self.assertEqual(
                result_url, 
                expected_url, 
                f"Failed for place: {place}\nExpected: {expected_url}\nGot: {result_url}"
            )

if __name__ == '__main__':
    unittest.main() 